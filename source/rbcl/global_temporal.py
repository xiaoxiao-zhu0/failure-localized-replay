"""Global-clock, multi-timescale steady-state replay for task-free OCL."""

from __future__ import annotations

import hashlib
import random
from typing import Optional, Sequence

import torch

from .causal_er_ace import CausalERACE


class GlobalTemporalSteadyERACE(CausalERACE):
    """ER-ACE with a time-only, multi-timescale memory policy.

    The policy deliberately has no input from model loss, gradients, current
    accuracy, task boundaries, labels, class counts, or sample features.  It
    partitions *memory time* rather than semantic classes: two deterministic
    renewal reservoirs represent short and medium horizons, while a global
    reservoir retains a uniform long-horizon anchor.  Both admission and
    replay quotas are fixed functions of the raw sample clock.
    """

    def __init__(
        self,
        *,
        renewal_periods: Sequence[int] = (512, 4096),
        seed: int = 0,
        **kwargs,
    ) -> None:
        if not renewal_periods or any(period <= 0 for period in renewal_periods):
            raise ValueError("renewal_periods must contain positive integers")
        super().__init__(seed=seed, memory_policy="reservoir", **kwargs)
        if self.mem_size < 2 * len(renewal_periods) + 1:
            raise ValueError("mem_size is too small for the requested time levels")

        self.renewal_periods = tuple(int(period) for period in renewal_periods)
        level_count = len(self.renewal_periods) + 1
        # The global anchor receives half the budget.  The remaining capacity
        # is divided exactly across finite renewal levels; this is fixed before
        # seeing the stream and is never adapted to it.
        global_capacity = self.mem_size // 2
        finite_capacity = self.mem_size - global_capacity
        base = finite_capacity // len(self.renewal_periods)
        remainder = finite_capacity - base * len(self.renewal_periods)
        capacities = [
            base + int(index < remainder)
            for index in range(len(self.renewal_periods))
        ] + [global_capacity]
        if sum(capacities) != self.mem_size or len(capacities) != level_count:
            raise RuntimeError("invalid temporal capacity allocation")

        self._temporal_levels: list[dict] = []
        for index, (period, capacity) in enumerate(
            zip(self.renewal_periods, capacities[:-1])
        ):
            self._temporal_levels.append(
                {
                    "period": period,
                    "capacity": capacity,
                    "window": -1,
                    "seen": 0,
                    "items": [],
                    "rng": random.Random(int(seed) + 6_100_001 + index),
                }
            )
        self._temporal_levels.append(
            {
                "period": None,
                "capacity": capacities[-1],
                "window": None,
                "seen": 0,
                "items": [],
                "rng": random.Random(int(seed) + 6_100_099),
            }
        )
        self._temporal_replay_rng = random.Random(int(seed) + 6_100_777)
        self._global_sample_clock = 0
        self._renewal_count = 0
        self._schedule_hasher = hashlib.sha256()
        self._replay_hasher = hashlib.sha256()

    @staticmethod
    def _copy_record(
        x: torch.Tensor,
        y: torch.Tensor,
        task_id: torch.Tensor,
        clock: int,
    ) -> tuple[torch.Tensor, int, int, int]:
        # Labels/task IDs are retained only so ER-ACE can train on a replayed
        # sample. They are never read by admission, eviction, or sampling.
        return (
            x.detach().cpu().clone(),
            int(y.item()),
            int(task_id.item()),
            int(clock),
        )

    def _refresh_global_memory_view(self) -> None:
        self._memory_x = []
        self._memory_y = []
        self._memory_tid = []
        for level in self._temporal_levels:
            for sample_x, label, task_id, _ in level["items"]:
                self._memory_x.append(sample_x)
                self._memory_y.append(label)
                self._memory_tid.append(task_id)

    def _renew_or_insert(
        self,
        level: dict,
        record: tuple[torch.Tensor, int, int, int],
        level_index: int,
    ) -> None:
        period: Optional[int] = level["period"]
        if period is not None:
            window = (self._global_sample_clock - 1) // period
            if window != level["window"]:
                level["window"] = window
                level["seen"] = 0
                level["items"] = []
                self._renewal_count += 1
                self._schedule_hasher.update(
                    f"renew:{self._global_sample_clock}:{level_index}:{window};".encode()
                )

        level["seen"] += 1
        replacement = -1
        if len(level["items"]) < level["capacity"]:
            level["items"].append(record)
            replacement = len(level["items"]) - 1
        else:
            candidate = level["rng"].randrange(level["seen"])
            if candidate < level["capacity"]:
                level["items"][candidate] = record
                replacement = candidate
        self._schedule_hasher.update(
            (
                f"insert:{self._global_sample_clock}:{level_index}:"
                f"{level['window']}:{replacement};"
            ).encode()
        )

    def _update_memory(
        self, x: torch.Tensor, y: torch.Tensor, task_ids: torch.Tensor
    ) -> None:
        for sample_x, sample_y, sample_tid in zip(x, y, task_ids):
            self._global_sample_clock += 1
            record = self._copy_record(
                sample_x, sample_y, sample_tid, self._global_sample_clock
            )
            for level_index, level in enumerate(self._temporal_levels):
                self._renew_or_insert(level, record, level_index)
        self._seen_samples = self._global_sample_clock
        self._refresh_global_memory_view()

    def _sample_memory(self):
        filled = [level for level in self._temporal_levels if level["items"]]
        if not filled:
            return None
        unique_clock_count = len(
            {
                item[3]
                for level in filled
                for item in level["items"]
            }
        )
        count = min(self.batch_size_mem, unique_clock_count)
        base = count // len(self._temporal_levels)
        remainder = count - base * len(self._temporal_levels)
        selected: list[tuple[torch.Tensor, int, int, int]] = []
        selected_clocks: set[int] = set()

        for level_index, level in enumerate(self._temporal_levels):
            requested = base + int(level_index < remainder)
            candidates = [
                item for item in level["items"] if item[3] not in selected_clocks
            ]
            chosen_count = min(requested, len(candidates))
            if chosen_count:
                chosen = self._temporal_replay_rng.sample(candidates, chosen_count)
                selected.extend(chosen)
                selected_clocks.update(item[3] for item in chosen)
                self._replay_hasher.update(
                    f"{self._global_sample_clock}:{level_index}:".encode()
                )
                self._replay_hasher.update(
                    ",".join(str(item[3]) for item in chosen).encode()
                )

        if len(selected) < count:
            candidates = [
                item
                for level in self._temporal_levels
                for item in level["items"]
                if item[3] not in selected_clocks
            ]
            fill = min(count - len(selected), len(candidates))
            if fill:
                selected.extend(self._temporal_replay_rng.sample(candidates, fill))

        if not selected:
            return None
        self._replay_examples += len(selected)
        x = torch.stack([item[0] for item in selected]).to(self.device)
        y = torch.tensor(
            [item[1] for item in selected], dtype=torch.long, device=self.device
        )
        task_ids = torch.tensor(
            [item[2] for item in selected], dtype=torch.long, device=self.device
        )
        return x, y, task_ids

    def rbcl_summary(self) -> dict:
        report = super().rbcl_summary()
        report["global_temporal_steady_memory"] = {
            "enabled": True,
            "raw_sample_clock": self._global_sample_clock,
            "renewal_periods": list(self.renewal_periods),
            "levels": [
                {
                    "period": level["period"],
                    "capacity": level["capacity"],
                    "filled": len(level["items"]),
                    "active_window": level["window"],
                    "seen_in_active_window": level["seen"],
                }
                for level in self._temporal_levels
            ],
            "renewal_count": self._renewal_count,
            "schedule_digest": self._schedule_hasher.hexdigest(),
            "replay_digest": self._replay_hasher.hexdigest(),
            "policy_uses_only_global_sample_clock": True,
            "policy_reads_model_state": False,
            "policy_reads_loss_or_gradient": False,
            "policy_reads_accuracy_or_validation": False,
            "policy_reads_task_id_for_decisions": False,
            "policy_reads_label_for_decisions": False,
            "fixed_uniform_time_level_replay": True,
            "duplicate_clock_replay_prevented": True,
        }
        return report


__all__ = ["GlobalTemporalSteadyERACE"]
