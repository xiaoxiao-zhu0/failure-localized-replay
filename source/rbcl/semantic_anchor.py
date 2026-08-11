"""Task-free fixed-hybrid replay with a global additive semantic anchor.

The anchor is deliberately not a replay controller: its strength and refresh
times are fixed in raw sample-clock units.  It never inspects loss, gradients,
validation performance, experience IDs, or the hard/blurry stream identity.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from .causal_er_ace import CausalERACE


# Historical name retained because the semantic-memory implementation imports
# it directly. The registry now covers every image benchmark used by RBCL.
_CIFAR_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "tinyimagenet": (
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010),
    ),
    "core50": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class GlobalSemanticAnchorERACE(CausalERACE):
    """Fixed hybrid ER-ACE with a past-only additive semantic teacher.

    ``A += zz^T`` and ``B += z onehot(y)`` are updated after an arriving
    minibatch has trained.  The teacher used for the next update is refreshed
    only every fixed number of raw arriving samples.  Thus the scheduling rule
    is identical under hard/blurry taskification of the same sample stream.
    """

    def __init__(
        self,
        *args,
        dataset_family: str,
        num_classes: int,
        anchor_lambda: float = 0.25,
        anchor_temperature: float = 2.0,
        anchor_refresh_samples: int = 512,
        anchor_ridge: float = 1.0,
        anchor_scope: str = "current",
        anchor_audit_only: bool = False,
        anchor_audit_every: int = 512,
        **kwargs,
    ):
        if dataset_family not in _CIFAR_STATS:
            raise ValueError(
                "semantic anchor supports CIFAR-10/100, Tiny ImageNet, and CORe50"
            )
        if anchor_lambda < 0.0:
            raise ValueError("anchor_lambda must be non-negative")
        if anchor_scope not in {"current", "replay"}:
            raise ValueError("anchor_scope must be current or replay")
        if anchor_temperature <= 0.0 or anchor_refresh_samples <= 0:
            raise ValueError("anchor temperature/refresh must be positive")
        super().__init__(*args, memory_policy="hybrid", **kwargs)
        self.dataset_family = dataset_family
        self.num_classes = int(num_classes)
        self.anchor_lambda = float(anchor_lambda)
        self.anchor_temperature = float(anchor_temperature)
        self.anchor_refresh_samples = int(anchor_refresh_samples)
        self.anchor_ridge = float(anchor_ridge)
        self.anchor_scope = anchor_scope
        self.anchor_audit_only = bool(anchor_audit_only)
        self.anchor_audit_every = int(anchor_audit_every)
        if self.anchor_audit_every <= 0:
            raise ValueError("anchor_audit_every must be positive")
        self._anchor_encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        self._anchor_encoder.fc = torch.nn.Identity()
        self._anchor_encoder.to(self.device).eval()
        for parameter in self._anchor_encoder.parameters():
            parameter.requires_grad_(False)
        self._anchor_dim = 513
        self._anchor_gram = torch.zeros(
            (self._anchor_dim, self._anchor_dim), dtype=torch.float64,
            device=self.device,
        )
        self._anchor_targets = torch.zeros(
            (self._anchor_dim, self.num_classes), dtype=torch.float64,
            device=self.device,
        )
        self._anchor_identity = torch.eye(
            self._anchor_dim, dtype=torch.float64, device=self.device
        )
        self._anchor_weights: Optional[torch.Tensor] = None
        self._anchor_seen_classes: set[int] = set()
        self._anchor_sample_clock = 0
        self._anchor_refreshes = 0
        self._anchor_loss_steps = 0
        self._anchor_audit_history: list[dict[str, float]] = []

    @torch.no_grad()
    def _anchor_features(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = _CIFAR_STATS[self.dataset_family]
        mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)
        image_mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        image_std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        image = (x.to(self.device) * std + mean).clamp(0.0, 1.0)
        image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
        features = self._anchor_encoder((image - image_mean) / image_std)
        return torch.cat(
            [features, torch.ones((features.shape[0], 1), device=self.device)], dim=1
        ).double()

    def _refresh_anchor(self) -> None:
        if not self._anchor_seen_classes:
            return
        self._anchor_weights = torch.linalg.solve(
            self._anchor_gram + self.anchor_ridge * self._anchor_identity,
            self._anchor_targets,
        )
        self._anchor_refreshes += 1

    def _add_auxiliary_loss(
        self, current_x, current_y, current_tid, replay, replay_output=None
    ) -> None:
        # The teacher is always past-only. Current scope prevents same-batch
        # leakage; replay scope instead repairs historical replay predictions
        # and never constrains a freshly arriving sample.
        if self._anchor_weights is None or not self._anchor_seen_classes:
            return
        seen = torch.tensor(
            sorted(self._anchor_seen_classes), device=self.device, dtype=torch.long
        )
        if self.anchor_scope == "replay":
            if replay is None or replay_output is None:
                return
            anchor_x = replay[0]
            student_logits = replay_output[:, seen]
        else:
            eligible = torch.isin(current_y.to(self.device), seen)
            if not bool(eligible.any()):
                return
            anchor_x = current_x[eligible]
            student_logits = self.mb_output[eligible][:, seen]
        with torch.no_grad():
            teacher_logits = self._anchor_features(anchor_x) @ self._anchor_weights
            teacher_logits = teacher_logits[:, seen].float()
            if (
                self.anchor_audit_only
                and self.anchor_scope == "replay"
                and self._anchor_sample_clock % self.anchor_audit_every
                < int(current_y.numel())
            ):
                labels = replay[1].to(self.device)
                teacher_predictions = seen[teacher_logits.argmax(dim=1)]
                student_predictions = seen[student_logits.argmax(dim=1)]
                self._anchor_audit_history.append({
                    "sample_clock": float(self._anchor_sample_clock),
                    "teacher_replay_accuracy": float((teacher_predictions == labels).float().mean()),
                    "student_replay_accuracy": float((student_predictions == labels).float().mean()),
                    "teacher_student_agreement": float((teacher_predictions == student_predictions).float().mean()),
                })
        if self.anchor_audit_only:
            return
        temperature = self.anchor_temperature
        anchor_loss = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature * temperature)
        self.loss += self.anchor_lambda * anchor_loss
        self._anchor_loss_steps += 1

    @torch.no_grad()
    def _after_anchor_update(self, current_x, current_y, current_tid) -> None:
        features = self._anchor_features(current_x)
        labels = current_y.to(self.device).long()
        self._anchor_gram.add_(features.T @ features)
        self._anchor_targets.add_(
            features.T @ F.one_hot(labels, num_classes=self.num_classes).double()
        )
        self._anchor_seen_classes.update(int(label) for label in labels.cpu().tolist())
        previous_clock = self._anchor_sample_clock
        self._anchor_sample_clock += int(labels.numel())
        if previous_clock // self.anchor_refresh_samples != self._anchor_sample_clock // self.anchor_refresh_samples:
            self._refresh_anchor()

    def rbcl_summary(self) -> dict:
        report = super().rbcl_summary()
        report["global_semantic_anchor"] = {
            "enabled": True,
            "dataset_family": self.dataset_family,
            "input_normalization_mean": list(_CIFAR_STATS[self.dataset_family][0]),
            "input_normalization_std": list(_CIFAR_STATS[self.dataset_family][1]),
            "frozen_imagenet_resnet18": True,
            "strictly_additive_statistics": True,
            "memory_policy": "fixed_hybrid",
            "anchor_lambda": self.anchor_lambda,
            "anchor_scope": self.anchor_scope,
            "anchor_temperature": self.anchor_temperature,
            "refresh_samples": self.anchor_refresh_samples,
            "raw_sample_clock": self._anchor_sample_clock,
            "refreshes": self._anchor_refreshes,
            "anchor_loss_steps": self._anchor_loss_steps,
            "audit_only": self.anchor_audit_only,
            "compatibility_audit": {
                "count": len(self._anchor_audit_history),
                "history": self._anchor_audit_history,
                **{key: (sum(row[key] for row in self._anchor_audit_history) / len(self._anchor_audit_history) if self._anchor_audit_history else 0.0) for key in ("teacher_replay_accuracy", "student_replay_accuracy", "teacher_student_agreement")},
            },
            "policy_uses_loss_gradient_validation_or_task_id": False,
        }
        return report


__all__ = ["GlobalSemanticAnchorERACE"]
