"""Causal minibatch-progressive ER-ACE control for protocol auditing."""

from __future__ import annotations

import copy
import hashlib
import math
import random
from pathlib import Path
from collections import Counter
from typing import Callable, List, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer

from avalanche.core import SupervisedPlugin
from avalanche.models.utils import avalanche_forward
from avalanche.training.plugins.evaluation import EvaluationPlugin, default_evaluator
from avalanche.training.regularization import cross_entropy_with_oh_targets
from avalanche.training.templates import SupervisedTemplate
from avalanche.training.templates.strategy_mixin_protocol import CriterionType
from avalanche.training.utils import get_last_fc_layer


class CausalERACE(SupervisedTemplate):
    """ER-ACE with causal, progressive memory insertion.

    Avalanche's built-in ER_ACE is explicitly adapted to a non-online
    experience protocol and preloads the entire current experience into its
    class-balanced buffer before optimization.  This control keeps the ACE
    objective but inserts each current minibatch only after its update during
    the first epoch, using either Reservoir or class-balanced Reservoir
    storage. It therefore never replays a future sample before arrival.
    """

    def __init__(
        self,
        *,
        model: Module,
        optimizer: Optimizer,
        criterion: CriterionType = CrossEntropyLoss(),
        mem_size: int = 200,
        batch_size_mem: int = 10,
        seed: int = 0,
        memory_policy: str = "reservoir",
        value_coverage_audit: bool = False,
        value_coverage_audit_every: int = 500,
        value_coverage_rho: float = 0.95,
        paired_update_audit: bool = False,
        paired_update_audit_every: int = 512,
        memory_trace_signature: bool = False,
        memory_trace_audit: bool = False,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: Optional[int] = 1,
        device: Union[str, torch.device] = "cpu",
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: Union[
            EvaluationPlugin, Callable[[], EvaluationPlugin]
        ] = default_evaluator,
        eval_every: int = -1,
        **kwargs,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **kwargs,
        )
        self.mem_size = int(mem_size)
        self.batch_size_mem = int(batch_size_mem)
        if memory_policy not in {"reservoir", "class_balanced", "hybrid"}:
            raise ValueError(
                "memory_policy must be reservoir, class_balanced, or hybrid"
            )
        self.memory_policy = memory_policy
        if value_coverage_audit_every <= 0:
            raise ValueError("value_coverage_audit_every must be positive")
        if not 0.0 < value_coverage_rho <= 1.0:
            raise ValueError("value_coverage_rho must be in (0, 1]")
        self.value_coverage_audit = bool(value_coverage_audit)
        self.value_coverage_audit_every = int(value_coverage_audit_every)
        self.value_coverage_rho = float(value_coverage_rho)
        self._rng = random.Random(int(seed))
        self._audit_rng = random.Random(int(seed) + 1_000_003)
        self._memory_x: list[torch.Tensor] = []
        self._memory_y: list[int] = []
        self._memory_tid: list[int] = []
        self._seen_samples = 0
        self._class_memory: dict[
            int, list[tuple[torch.Tensor, int]]
        ] = {}
        self._class_seen: dict[int, int] = {}
        self._hybrid_memory_x: list[torch.Tensor] = []
        self._hybrid_memory_y: list[int] = []
        self._hybrid_memory_tid: list[int] = []
        self._hybrid_seen = 0
        self._epoch_index = 0
        self._replay_examples = 0
        self._observed_classes: set[int] = set()
        self._value_coverage_audit_history: list[dict[str, float]] = []
        if paired_update_audit_every <= 0:
            raise ValueError("paired_update_audit_every must be positive")
        self.paired_update_audit = bool(paired_update_audit)
        self.paired_update_audit_every = int(paired_update_audit_every)
        self._paired_update_audit_records: list[dict[str, object]] = []
        self.memory_trace_audit = bool(memory_trace_audit)
        self.memory_trace_signature = bool(
            memory_trace_signature or memory_trace_audit
        )
        self._memory_trace_history: list[dict[str, object]] = []
        self._memory_trace_replay_hash = hashlib.sha256()
        self._memory_trace_global_class_draws: Counter[int] = Counter()
        self._memory_trace_global_source_draws: Counter[str] = Counter()
        self._memory_arrival: dict[str, int] = {}
        self._reset_memory_trace_experience()

    def _before_training_exp(self, **kwargs):
        self._epoch_index = 0
        self._reset_memory_trace_experience()
        super()._before_training_exp(**kwargs)

    def _after_training_exp(self, **kwargs):
        if self.memory_trace_audit:
            self._memory_trace_history.append(self._memory_trace_snapshot())
        super()._after_training_exp(**kwargs)

    def _reset_memory_trace_experience(self) -> None:
        self._memory_trace_exp_class_draws: Counter[int] = Counter()
        self._memory_trace_exp_source_draws: Counter[str] = Counter()
        self._memory_trace_exp_ages: list[int] = []

    @staticmethod
    def _memory_trace_key(sample: torch.Tensor, label: int) -> str:
        payload = sample.detach().cpu().contiguous().numpy().tobytes()
        return f"{int(label)}:{hashlib.sha1(payload).hexdigest()}"

    def _memory_source_for_index(self, index: int) -> str:
        if self.memory_policy == "reservoir":
            return "reservoir"
        semantic_size = sum(len(items) for items in self._class_memory.values())
        return "semantic" if index < semantic_size else "reservoir"

    def _refresh_memory_trace_arrivals(self) -> None:
        if not self.memory_trace_audit:
            return
        clock = max(self._seen_samples, self._hybrid_seen)
        present = {
            self._memory_trace_key(sample, label)
            for sample, label in zip(self._memory_x, self._memory_y)
        }
        self._memory_arrival = {
            key: self._memory_arrival.get(key, clock) for key in present
        }

    def _record_memory_trace_replay(self, indices: list[int]) -> None:
        if not self.memory_trace_signature:
            return
        experience = int(
            getattr(getattr(self, "experience", None), "current_experience", -1)
        )
        labels = [int(self._memory_y[index]) for index in indices]
        sources = [self._memory_source_for_index(index) for index in indices]
        signature = (
            f"{experience}|{','.join(map(str, indices))}|"
            f"{','.join(map(str, labels))}|{','.join(sources)}\n"
        )
        self._memory_trace_replay_hash.update(signature.encode("ascii"))
        if not self.memory_trace_audit:
            return
        clock = max(self._seen_samples, self._hybrid_seen)
        for index, label, source in zip(indices, labels, sources):
            self._memory_trace_exp_class_draws[label] += 1
            self._memory_trace_exp_source_draws[source] += 1
            self._memory_trace_global_class_draws[label] += 1
            self._memory_trace_global_source_draws[source] += 1
            key = self._memory_trace_key(self._memory_x[index], label)
            arrival = self._memory_arrival.get(key, clock)
            self._memory_trace_exp_ages.append(max(0, clock - arrival))

    @staticmethod
    def _memory_trace_distribution(values: list[int]) -> dict[str, float]:
        if not values:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0}
        ordered = sorted(values)

        def percentile(fraction: float) -> float:
            index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
            return float(ordered[index])

        return {
            "count": len(ordered),
            "mean": float(sum(ordered) / len(ordered)),
            "p50": percentile(0.50),
            "p90": percentile(0.90),
        }

    @staticmethod
    def _memory_trace_count_stats(counts: Counter[int]) -> dict[str, float]:
        total = sum(counts.values())
        if total <= 0:
            return {"class_coverage": 0, "entropy": 0.0, "ess": 0.0}
        probabilities = [count / total for count in counts.values() if count > 0]
        entropy = -sum(value * math.log(value) for value in probabilities)
        ess = total * total / sum(count * count for count in counts.values())
        return {
            "class_coverage": len(counts),
            "entropy": float(entropy),
            "ess": float(ess),
        }

    def _memory_state_hash(self) -> str:
        digest = hashlib.sha256()
        for index, (sample, label, task_id) in enumerate(
            zip(self._memory_x, self._memory_y, self._memory_tid)
        ):
            digest.update(
                f"{index}|{self._memory_source_for_index(index)}|{label}|{task_id}|".encode(
                    "ascii"
                )
            )
            digest.update(sample.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _model_state_hash(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _memory_trace_snapshot(self) -> dict[str, object]:
        semantic_counts = Counter(
            label
            for label, samples in self._class_memory.items()
            for _ in samples
        )
        reservoir_labels = (
            self._memory_y
            if self.memory_policy == "reservoir"
            else self._hybrid_memory_y
        )
        reservoir_counts = Counter(int(label) for label in reservoir_labels)
        memory_counts = Counter(int(label) for label in self._memory_y)
        return {
            "experience": int(
                getattr(getattr(self, "experience", None), "current_experience", -1)
            ),
            "seen_samples": int(self._seen_samples),
            "memory_size": len(self._memory_x),
            "observed_classes": sorted(self._observed_classes),
            "semantic_size": sum(semantic_counts.values()),
            "reservoir_size": sum(reservoir_counts.values()),
            "semantic_class_counts": dict(sorted(semantic_counts.items())),
            "reservoir_class_counts": dict(sorted(reservoir_counts.items())),
            "memory_class_counts": dict(sorted(memory_counts.items())),
            "replay_class_draw_counts": dict(
                sorted(self._memory_trace_exp_class_draws.items())
            ),
            "replay_source_draw_counts": dict(
                sorted(self._memory_trace_exp_source_draws.items())
            ),
            "replay_age_samples": self._memory_trace_distribution(
                self._memory_trace_exp_ages
            ),
            "replay_distribution": self._memory_trace_count_stats(
                self._memory_trace_exp_class_draws
            ),
            "memory_state_hash": self._memory_state_hash(),
        }

    def _sample_memory(self):
        if not self._memory_x:
            return None
        count = min(self.batch_size_mem, len(self._memory_x))
        indices = self._rng.sample(range(len(self._memory_x)), count)
        self._last_replay_memory_indices = list(indices)
        self._record_memory_trace_replay(indices)
        x = torch.stack([self._memory_x[index] for index in indices]).to(self.device)
        y = torch.tensor(
            [self._memory_y[index] for index in indices],
            dtype=torch.long,
            device=self.device,
        )
        task_ids = torch.tensor(
            [self._memory_tid[index] for index in indices],
            dtype=torch.long,
            device=self.device,
        )
        self._replay_examples += count
        return x, y, task_ids

    def _update_reservoir(
        self, x: torch.Tensor, y: torch.Tensor, task_ids: torch.Tensor
    ) -> None:
        for sample_x, sample_y, sample_tid in zip(x, y, task_ids):
            self._seen_samples += 1
            cpu_x = sample_x.detach().cpu().clone()
            label = int(sample_y.item())
            task_id = int(sample_tid.item())
            if len(self._memory_x) < self.mem_size:
                self._memory_x.append(cpu_x)
                self._memory_y.append(label)
                self._memory_tid.append(task_id)
                continue
            replacement = self._rng.randrange(self._seen_samples)
            if replacement < self.mem_size:
                self._memory_x[replacement] = cpu_x
                self._memory_y[replacement] = label
                self._memory_tid[replacement] = task_id

    def _class_capacities(self) -> dict[int, int]:
        classes = sorted(self._class_memory)
        if not classes:
            return {}
        class_capacity = (
            self.mem_size // 2
            if self.memory_policy == "hybrid"
            else self.mem_size
        )
        base = class_capacity // len(classes)
        remainder = class_capacity - base * len(classes)
        return {
            label: base + int(index < remainder)
            for index, label in enumerate(classes)
        }

    def _rebalance_class_memory(self) -> None:
        capacities = self._class_capacities()
        for label, samples in self._class_memory.items():
            capacity = capacities[label]
            if len(samples) > capacity:
                self._class_memory[label] = self._rng.sample(samples, capacity)

    def _refresh_flat_memory(self) -> None:
        self._memory_x = []
        self._memory_y = []
        self._memory_tid = []
        for label in sorted(self._class_memory):
            for sample_x, task_id in self._class_memory[label]:
                self._memory_x.append(sample_x)
                self._memory_y.append(label)
                self._memory_tid.append(task_id)
        if self.memory_policy == "hybrid":
            self._memory_x.extend(self._hybrid_memory_x)
            self._memory_y.extend(self._hybrid_memory_y)
            self._memory_tid.extend(self._hybrid_memory_tid)

    def _update_class_balanced(
        self, x: torch.Tensor, y: torch.Tensor, task_ids: torch.Tensor
    ) -> None:
        for sample_x, sample_y, sample_tid in zip(x, y, task_ids):
            label = int(sample_y.item())
            task_id = int(sample_tid.item())
            if label not in self._class_memory:
                self._class_memory[label] = []
                self._class_seen[label] = 0
                self._rebalance_class_memory()

            self._seen_samples += 1
            self._class_seen[label] += 1
            capacity = self._class_capacities()[label]
            if capacity == 0:
                continue

            samples = self._class_memory[label]
            item = (sample_x.detach().cpu().clone(), task_id)
            if len(samples) < capacity:
                samples.append(item)
                continue
            replacement = self._rng.randrange(self._class_seen[label])
            if replacement < capacity:
                samples[replacement] = item
        self._refresh_flat_memory()

    def _update_hybrid_reservoir(
        self, x: torch.Tensor, y: torch.Tensor, task_ids: torch.Tensor
    ) -> None:
        capacity = self.mem_size - self.mem_size // 2
        for sample_x, sample_y, sample_tid in zip(x, y, task_ids):
            self._hybrid_seen += 1
            cpu_x = sample_x.detach().cpu().clone()
            label = int(sample_y.item())
            task_id = int(sample_tid.item())
            if len(self._hybrid_memory_x) < capacity:
                self._hybrid_memory_x.append(cpu_x)
                self._hybrid_memory_y.append(label)
                self._hybrid_memory_tid.append(task_id)
                continue
            replacement = self._rng.randrange(self._hybrid_seen)
            if replacement < capacity:
                self._hybrid_memory_x[replacement] = cpu_x
                self._hybrid_memory_y[replacement] = label
                self._hybrid_memory_tid[replacement] = task_id

    def _update_memory(
        self, x: torch.Tensor, y: torch.Tensor, task_ids: torch.Tensor
    ) -> None:
        if self.memory_policy == "class_balanced":
            self._update_class_balanced(x, y, task_ids)
        elif self.memory_policy == "hybrid":
            self._update_class_balanced(x, y, task_ids)
            self._update_hybrid_reservoir(x, y, task_ids)
            self._refresh_flat_memory()
        else:
            self._update_reservoir(x, y, task_ids)

    def _before_replay_selection(
        self,
        current_x: torch.Tensor,
        current_y: torch.Tensor,
        current_tid: torch.Tensor,
    ) -> None:
        """Optional causal hook for replay-policy controllers.

        The base strategy deliberately does nothing. Subclasses may inspect
        the already-arrived current minibatch and the existing memory before
        replay is sampled, but must not mutate the training model in this
        hook.
        """

    def _after_anchor_update(
        self,
        current_x: torch.Tensor,
        current_y: torch.Tensor,
        current_tid: torch.Tensor,
    ) -> None:
        """Optional post-anchor hook; the base strategy does nothing."""

    def _after_memory_update(
        self,
        current_x: torch.Tensor,
        current_y: torch.Tensor,
        current_tid: torch.Tensor,
    ) -> None:
        """Optional hook after first-pass causal memory insertion."""

    def _after_memory_population_stage(
        self,
        current_x: torch.Tensor,
        current_y: torch.Tensor,
        current_tid: torch.Tensor,
    ) -> None:
        """Optional hook after the memory-population stage of every update.

        On the first pass this runs after the arrived minibatch has been
        inserted into memory.  On later local epochs the population stage is
        an explicit no-op, but the hook still runs.  The separation is useful
        for faithful wrappers such as OBC whose classifier-only update occurs
        after the base learner's train-and-populate steps.
        """

    def _add_auxiliary_loss(
        self,
        current_x: torch.Tensor,
        current_y: torch.Tensor,
        current_tid: torch.Tensor,
        replay: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        replay_output: torch.Tensor | None = None,
    ) -> None:
        """Optionally add a causal auxiliary objective to ``self.loss``.

        The default is intentionally a no-op.  It gives derived strategies a
        narrow extension point without changing the ER-ACE/replay path used
        by all existing controls.
        """

    def _record_paired_update_audit(
        self,
        current_loss: torch.Tensor,
        replay_loss: torch.Tensor | None,
        current_count: int,
    ) -> None:
        """Record paired gradient geometry without changing training."""
        if not self.paired_update_audit or replay_loss is None:
            return
        next_clock = self._seen_samples + int(current_count)
        if next_clock <= 0 or next_clock % self.paired_update_audit_every:
            return
        _, head = get_last_fc_layer(self.model)
        weight = getattr(head, "weight", None)
        if weight is None or not weight.requires_grad:
            return

        named_parameters = tuple(
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        )
        parameters = tuple(parameter for _, parameter in named_parameters)
        current_gradients = torch.autograd.grad(
            current_loss, parameters, retain_graph=True, allow_unused=True
        )
        replay_gradients = torch.autograd.grad(
            replay_loss, parameters, retain_graph=True, allow_unused=True
        )
        weight_index = next(
            (
                index
                for index, (_, parameter) in enumerate(named_parameters)
                if parameter is weight
            ),
            None,
        )
        if weight_index is None:
            return

        current_gradient = current_gradients[weight_index]
        replay_gradient = replay_gradients[weight_index]
        if current_gradient is None or replay_gradient is None:
            return

        head_parameter_ids = {id(parameter) for parameter in head.parameters()}

        def geometry(indices: list[int]) -> dict[str, float | bool | int | None]:
            dot = torch.zeros((), device=self.device)
            current_sq = torch.zeros((), device=self.device)
            replay_sq = torch.zeros((), device=self.device)
            tensor_conflicts = 0
            tensor_pairs = 0
            for index in indices:
                current = current_gradients[index]
                replay = replay_gradients[index]
                if current is not None:
                    current_sq += current.detach().square().sum()
                if replay is not None:
                    replay_sq += replay.detach().square().sum()
                if current is not None and replay is not None:
                    tensor_dot = (current.detach() * replay.detach()).sum()
                    dot += tensor_dot
                    tensor_pairs += 1
                    tensor_conflicts += int(float(tensor_dot) < 0.0)
            current_norm = current_sq.sqrt()
            replay_norm = replay_sq.sqrt()
            total_sq = current_sq + replay_sq + 2.0 * dot
            total_norm = total_sq.clamp_min(0.0).sqrt()
            current_total = current_sq + dot
            replay_total = replay_sq + dot
            current_denom = current_norm * total_norm
            replay_denom = replay_norm * total_norm
            pair_denom = current_norm * replay_norm
            return {
                "current_replay_cosine": (
                    float((dot / pair_denom).detach())
                    if float(pair_denom.detach()) > 1e-12
                    else None
                ),
                "current_total_alignment_cosine": (
                    float((current_total / current_denom).detach())
                    if float(current_denom.detach()) > 1e-12
                    else None
                ),
                "replay_total_alignment_cosine": (
                    float((replay_total / replay_denom).detach())
                    if float(replay_denom.detach()) > 1e-12
                    else None
                ),
                "current_gradient_norm": float(current_norm.detach()),
                "replay_gradient_norm": float(replay_norm.detach()),
                "replay_to_current_norm_ratio": (
                    float((replay_norm / current_norm).detach())
                    if float(current_norm.detach()) > 1e-12
                    else None
                ),
                "current_harmful_first_order": bool(float(current_total.detach()) <= 0.0),
                "replay_harmful_first_order": bool(float(replay_total.detach()) <= 0.0),
                "parameter_tensor_conflict_fraction": (
                    tensor_conflicts / tensor_pairs if tensor_pairs else 0.0
                ),
                "parameter_tensor_pairs": tensor_pairs,
            }

        all_indices = list(range(len(named_parameters)))
        head_indices = [
            index
            for index, (_, parameter) in enumerate(named_parameters)
            if id(parameter) in head_parameter_ids
        ]
        backbone_indices = [
            index for index in all_indices if index not in set(head_indices)
        ]
        self._paired_update_audit_records.append(
            {
                "sample_clock": int(next_clock),
                "current_weight_gradient": current_gradient.detach().cpu(),
                "replay_weight_gradient": replay_gradient.detach().cpu(),
                "geometry": {
                    "model": geometry(all_indices),
                    "backbone": geometry(backbone_indices),
                    "classifier": geometry(head_indices),
                },
            }
        )

    def export_paired_update_audit(self, output_path: str | Path) -> dict[str, object]:
        """Persist audit tensors outside the JSON experiment summary."""
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "stage": "paired update geometry audit",
                "format_version": 2,
                "sample_clock_interval": self.paired_update_audit_every,
                "training_objective_modified": False,
                "records": self._paired_update_audit_records,
            },
            destination,
        )
        return {"path": str(destination), "record_count": len(self._paired_update_audit_records)}

    @staticmethod
    def _gradient_vector(
        loss: torch.Tensor, parameters: tuple[torch.nn.Parameter, ...]
    ) -> torch.Tensor:
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        parts = []
        for parameter, gradient in zip(parameters, gradients):
            if gradient is None:
                parts.append(torch.zeros_like(parameter).reshape(-1))
            else:
                parts.append(gradient.detach().reshape(-1))
        return torch.cat(parts)

    def _head_step_copy(self, vector: torch.Tensor):
        candidate = copy.deepcopy(self.model).to(self.device)
        _, head = get_last_fc_layer(candidate)
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        offset = 0
        with torch.no_grad():
            for parameter in head.parameters():
                if not parameter.requires_grad:
                    continue
                width = parameter.numel()
                parameter.add_(
                    vector[offset : offset + width].reshape_as(parameter),
                    alpha=-learning_rate,
                )
                offset += width
        candidate.eval()
        return candidate

    @staticmethod
    def _ace_current_loss(
        logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        current_classes = torch.unique(labels)
        one_hot = F.one_hot(
            labels, num_classes=logits.shape[1]
        )[:, current_classes]
        return cross_entropy_with_oh_targets(
            logits[:, current_classes], one_hot
        )

    def _record_value_coverage_audit(
        self,
        current_x: torch.Tensor,
        current_y: torch.Tensor,
        current_tid: torch.Tensor,
    ) -> None:
        """Audit value-aware coverage routing on classifier-head copies only."""
        if not self.value_coverage_audit:
            return
        iteration = int(self.clock.train_iterations) + 1
        if iteration % self.value_coverage_audit_every != 0:
            return
        if len(self._memory_x) < 16 or int(current_y.numel()) < 8:
            return

        current_indices = list(range(int(current_y.numel())))
        self._audit_rng.shuffle(current_indices)
        current_cut = max(1, len(current_indices) // 2)
        current_train_indices = current_indices[:current_cut]
        current_probe_indices = current_indices[current_cut:]
        if not current_probe_indices:
            return

        memory_indices = list(range(len(self._memory_x)))
        self._audit_rng.shuffle(memory_indices)
        memory_cut = max(8, int(len(memory_indices) * 0.75))
        candidate_indices = memory_indices[:memory_cut]
        probe_indices = memory_indices[memory_cut:]
        if not probe_indices:
            return

        candidate_x = torch.stack(
            [self._memory_x[index] for index in candidate_indices]
        ).to(self.device)
        candidate_y = torch.tensor(
            [self._memory_y[index] for index in candidate_indices],
            dtype=torch.long,
            device=self.device,
        )
        candidate_tid = torch.tensor(
            [self._memory_tid[index] for index in candidate_indices],
            dtype=torch.long,
            device=self.device,
        )
        memory_probe_x = torch.stack(
            [self._memory_x[index] for index in probe_indices]
        ).to(self.device)
        memory_probe_y = torch.tensor(
            [self._memory_y[index] for index in probe_indices],
            dtype=torch.long,
            device=self.device,
        )
        memory_probe_tid = torch.tensor(
            [self._memory_tid[index] for index in probe_indices],
            dtype=torch.long,
            device=self.device,
        )
        current_train_x = current_x[current_train_indices].to(self.device)
        current_train_y = current_y[current_train_indices].to(self.device)
        current_train_tid = current_tid[current_train_indices].to(self.device)
        current_probe_x = current_x[current_probe_indices].to(self.device)
        current_probe_y = current_y[current_probe_indices].to(self.device)
        current_probe_tid = current_tid[current_probe_indices].to(self.device)

        was_training = self.model.training
        self.model.eval()
        try:
            _, head = get_last_fc_layer(self.model)
            parameters = tuple(
                parameter
                for parameter in head.parameters()
                if parameter.requires_grad
            )
            current_logits = avalanche_forward(
                self.model, current_train_x, current_train_tid
            )
            current_loss = self._ace_current_loss(
                current_logits, current_train_y
            )
            current_gradient = self._gradient_vector(
                current_loss, parameters
            )

            candidate_logits = avalanche_forward(
                self.model, candidate_x, candidate_tid
            )
            candidate_losses = F.cross_entropy(
                candidate_logits, candidate_y, reduction="none"
            )
            memory_probe_logits = avalanche_forward(
                self.model, memory_probe_x, memory_probe_tid
            )
            memory_probe_loss = F.cross_entropy(
                memory_probe_logits, memory_probe_y
            )
            memory_probe_gradient = self._gradient_vector(
                memory_probe_loss, parameters
            )
            sample_gradients = [
                self._gradient_vector(loss, parameters)
                for loss in candidate_losses
            ]

            counts = Counter(self._memory_y)
            seen_classes = max(1, len(self._observed_classes))
            target = self.mem_size / seen_classes
            values = [
                float(torch.dot(gradient, memory_probe_gradient).cpu())
                for gradient in sample_gradients
            ]
            compatibilities = [
                float(torch.dot(gradient, current_gradient).cpu())
                for gradient in sample_gradients
            ]
            deficits = [
                max(0.0, target - counts[int(label)])
                for label in candidate_y.detach().cpu().tolist()
            ]
            select_count = min(
                max(8, self.batch_size_mem // 2),
                len(sample_gradients),
            )
            uniform_selection = self._audit_rng.sample(
                range(len(sample_gradients)), select_count
            )
            coverage_selection = sorted(
                range(len(sample_gradients)),
                key=lambda index: (deficits[index], values[index]),
                reverse=True,
            )[:select_count]
            value_selection = sorted(
                range(len(sample_gradients)),
                key=lambda index: values[index],
                reverse=True,
            )[:select_count]
            # Start from the exact Uniform selection. Replace a selected
            # sample only when an unselected candidate has a larger coverage
            # deficit while weakly dominating it in both first-order memory
            # value and current-update compatibility. Each swap is therefore
            # parameter-free and Pareto-safe under the audit proxies.
            routed_selection = list(uniform_selection)
            pareto_swap_count = 0
            while True:
                selected_set = set(routed_selection)
                best_swap = None
                best_key = None
                for candidate_index in range(len(sample_gradients)):
                    if candidate_index in selected_set:
                        continue
                    for selected_position, selected_index in enumerate(
                        routed_selection
                    ):
                        if deficits[candidate_index] <= deficits[selected_index]:
                            continue
                        if (
                            values[candidate_index] + 1e-12
                            < values[selected_index]
                        ):
                            continue
                        if (
                            compatibilities[candidate_index] + 1e-12
                            < compatibilities[selected_index]
                        ):
                            continue
                        key = (
                            deficits[candidate_index] - deficits[selected_index],
                            values[candidate_index] - values[selected_index],
                            compatibilities[candidate_index]
                            - compatibilities[selected_index],
                        )
                        if best_key is None or key > best_key:
                            best_key = key
                            best_swap = (
                                selected_position,
                                candidate_index,
                            )
                if best_swap is None:
                    break
                routed_selection[best_swap[0]] = best_swap[1]
                pareto_swap_count += 1

            with torch.no_grad():
                current_before = F.cross_entropy(
                    avalanche_forward(
                        self.model, current_probe_x, current_probe_tid
                    ),
                    current_probe_y,
                )
            current_only_model = self._head_step_copy(current_gradient)
            with torch.no_grad():
                current_only_ce = F.cross_entropy(
                    avalanche_forward(
                        current_only_model,
                        current_probe_x,
                        current_probe_tid,
                    ),
                    current_probe_y,
                )
                current_only_memory_ce = F.cross_entropy(
                    avalanche_forward(
                        current_only_model,
                        memory_probe_x,
                        memory_probe_tid,
                    ),
                    memory_probe_y,
                )
            current_only_progress = float(
                (current_before - current_only_ce).cpu()
            )

            policy_results: dict[str, dict[str, float]] = {}
            for policy_name, selection in (
                ("uniform", uniform_selection),
                ("coverage", coverage_selection),
                ("value", value_selection),
                ("value_coverage", routed_selection),
            ):
                replay_gradient = torch.stack(
                    [sample_gradients[index] for index in selection]
                ).mean(dim=0)
                policy_model = self._head_step_copy(
                    (current_gradient + replay_gradient) / 2.0
                )
                with torch.no_grad():
                    policy_current_ce = F.cross_entropy(
                        avalanche_forward(
                            policy_model,
                            current_probe_x,
                            current_probe_tid,
                        ),
                        current_probe_y,
                    )
                    policy_memory_ce = F.cross_entropy(
                        avalanche_forward(
                            policy_model,
                            memory_probe_x,
                            memory_probe_tid,
                        ),
                        memory_probe_y,
                    )
                progress = float(
                    (current_before - policy_current_ce).cpu()
                )
                memory_gain = float(
                    (current_only_memory_ce - policy_memory_ce).cpu()
                )
                selected_labels = {
                    int(candidate_y[index].item()) for index in selection
                }
                policy_results[policy_name] = {
                    "progress": progress,
                    "memory_gain": memory_gain,
                    "selected_undercovered_fraction": (
                        sum(deficits[index] > 0.0 for index in selection)
                        / len(selection)
                    ),
                    "selected_unique_classes": float(len(selected_labels)),
                }
                del policy_model

            uniform = policy_results["uniform"]
            routed = policy_results["value_coverage"]
            positive_uniform_progress = uniform["progress"] > 0.0
            self._value_coverage_audit_history.append(
                {
                    "experience": float(
                        getattr(self.experience, "current_experience", -1)
                    ),
                    "iteration": float(iteration),
                    "memory_size": float(len(self._memory_x)),
                    "seen_classes": float(seen_classes),
                    "undercovered_class_count": float(
                        sum(
                            counts.get(label, 0) < target
                            for label in self._observed_classes
                        )
                    ),
                    "memory_class_coverage": float(
                        len(counts) / seen_classes
                    ),
                    "pareto_swap_count": float(pareto_swap_count),
                    "positive_current_only_progress": float(
                        current_only_progress > 0.0
                    ),
                    "current_only_progress": current_only_progress,
                    **{
                        f"{policy}_{metric}": value
                        for policy, result in policy_results.items()
                        for metric, value in result.items()
                    },
                    "route_keeps_uniform_progress": float(
                        positive_uniform_progress
                        and routed["progress"]
                        >= self.value_coverage_rho * uniform["progress"]
                    ),
                    "route_beats_uniform_memory_gain": float(
                        routed["memory_gain"] >= uniform["memory_gain"]
                    ),
                    "route_pareto_vs_uniform": float(
                        positive_uniform_progress
                        and routed["progress"]
                        >= self.value_coverage_rho * uniform["progress"]
                        and routed["memory_gain"] >= uniform["memory_gain"]
                    ),
                    "route_beats_coverage_memory_gain": float(
                        routed["memory_gain"]
                        >= policy_results["coverage"]["memory_gain"]
                    ),
                    "route_beats_value_class_coverage": float(
                        routed["selected_undercovered_fraction"]
                        >= policy_results["value"][
                            "selected_undercovered_fraction"
                        ]
                    ),
                }
            )
            del current_only_model
        finally:
            self.model.train(was_training)

    def training_epoch(self, **kwargs):
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break

            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)
            current_x = self.mb_x.detach().clone()
            current_y = self.mb_y.detach().clone()
            current_tid = self.mb_task_id.detach().clone()
            self._observed_classes.update(
                int(label) for label in current_y.detach().cpu().tolist()
            )
            self._record_value_coverage_audit(
                current_x, current_y, current_tid
            )
            self._before_replay_selection(
                current_x, current_y, current_tid
            )
            replay = self._sample_memory()

            self.optimizer.zero_grad()
            self.loss = self._make_empty_loss()
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)

            if replay is None:
                self.loss += F.cross_entropy(self.mb_output, self.mb_y)
            else:
                replay_x, replay_y, replay_tid = replay
                replay_output = avalanche_forward(
                    self.model, replay_x, replay_tid
                )
                current_classes = torch.unique(self.mb_y)
                one_hot = F.one_hot(
                    self.mb_y, num_classes=self.mb_output.shape[1]
                )[:, current_classes]
                current_loss = cross_entropy_with_oh_targets(
                    self.mb_output[:, current_classes], one_hot
                )
                replay_loss = F.cross_entropy(replay_output, replay_y)
                self.loss += (current_loss + replay_loss) / 2.0
                if self._epoch_index == 0:
                    self._record_paired_update_audit(
                        current_loss, replay_loss, int(current_y.numel())
                    )

            self._auxiliary_current_loss = (
                current_loss if replay is not None else self.loss
            )
            self._auxiliary_replay_loss = replay_loss if replay is not None else None
            try:
                self._add_auxiliary_loss(
                    current_x,
                    current_y,
                    current_tid,
                    replay,
                    replay_output if replay is not None else None,
                )
            finally:
                self._auxiliary_current_loss = None
                self._auxiliary_replay_loss = None

            self._before_backward(**kwargs)
            self.backward()
            self._after_backward(**kwargs)
            self._before_update(**kwargs)
            self.optimizer_step()
            self._after_anchor_update(
                current_x, current_y, current_tid
            )
            self._after_update(**kwargs)

            # Each raw sample enters memory once, progressively in the first
            # epoch. Later epochs never add duplicate copies.
            if self._epoch_index == 0:
                self._update_memory(current_x, current_y, current_tid)
                self._refresh_memory_trace_arrivals()
                self._after_memory_update(current_x, current_y, current_tid)
            self._after_memory_population_stage(
                current_x, current_y, current_tid
            )
            self._after_training_iteration(**kwargs)
        self._epoch_index += 1

    def rbcl_summary(self) -> dict:
        audit_history = self._value_coverage_audit_history
        audit_keys = (
            "route_keeps_uniform_progress",
            "route_beats_uniform_memory_gain",
            "route_pareto_vs_uniform",
            "route_beats_coverage_memory_gain",
            "route_beats_value_class_coverage",
        )
        report = {
            "causal_progressive_insertion": True,
            "preloads_current_experience": False,
            "memory_policy": self.memory_policy,
            "memory_size": len(self._memory_x),
            "seen_samples": self._seen_samples,
            "replay_examples": self._replay_examples,
            "value_coverage_audit": {
                "enabled": self.value_coverage_audit,
                "every": self.value_coverage_audit_every,
                "rho": self.value_coverage_rho,
                "count": len(audit_history),
                **{
                    key: (
                        sum(item[key] for item in audit_history)
                        / len(audit_history)
                        if audit_history
                        else 0.0
                    )
                    for key in audit_keys
                },
                "history": audit_history,
            },
            "paired_update_audit": {
                "enabled": self.paired_update_audit,
                "every": self.paired_update_audit_every,
                "record_count": len(self._paired_update_audit_records),
                "training_objective_modified": False,
            },
        }
        if self.memory_trace_signature:
            report["memory_trace_determinism"] = {
                "enabled": True,
                "final_memory_hash": self._memory_state_hash(),
                "replay_index_hash": self._memory_trace_replay_hash.hexdigest(),
                "final_model_hash": self._model_state_hash(),
            }
        report["memory_trace_audit"] = {
            "enabled": self.memory_trace_audit,
            "training_objective_modified": False,
            "history": self._memory_trace_history,
            "global_replay_class_draw_counts": dict(
                sorted(self._memory_trace_global_class_draws.items())
            ),
            "global_replay_source_draw_counts": dict(
                sorted(self._memory_trace_global_source_draws.items())
            ),
        }
        return report


__all__ = ["CausalERACE"]
