"""Controlled replay-memory corruption audits (not a proposed method)."""

from __future__ import annotations

from typing import Dict

import torch

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


class ReplayLabelNoisePlugin(SupervisedPlugin):
    """Corrupt only replay-suffix labels to test memory-quality sensitivity.

    The current batch is never modified. Replacement labels are drawn by a
    cyclic shift inside the replay suffix, so the corruption remains within
    labels already represented by the batch and does not require test data.
    """

    def __init__(self, noise_rate: float, seed: int):
        super().__init__()
        if not 0.0 <= noise_rate <= 1.0:
            raise ValueError("noise_rate must be in [0, 1]")
        self.noise_rate = float(noise_rate)
        self.generator = torch.Generator().manual_seed(int(seed))
        self.corrupted_examples = 0
        self.replay_examples = 0

    def before_forward(self, strategy, **kwargs):
        if self.noise_rate == 0.0:
            return
        batch_sizes = getattr(getattr(strategy, "dataloader", None), "batch_sizes", ())
        if len(batch_sizes) < 2:
            return
        current_n = int(batch_sizes[0])
        targets = strategy.mb_y.long()
        if current_n <= 0 or current_n >= targets.numel():
            return
        memory_targets = targets[current_n:]
        if memory_targets.numel() < 2:
            return
        mask = torch.rand(memory_targets.numel(), generator=self.generator) < self.noise_rate
        if not bool(mask.any()):
            return
        updated = targets.clone()
        replacements = torch.roll(memory_targets, shifts=1)
        device_mask = mask.to(memory_targets.device)
        updated_memory = memory_targets.clone()
        updated_memory[device_mask] = replacements[device_mask]
        updated[current_n:] = updated_memory
        strategy.mbatch[1] = updated
        self.corrupted_examples += int(mask.sum().item())
        self.replay_examples += int(memory_targets.numel())

    def summary(self) -> Dict[str, float]:
        return {
            "noise_rate": self.noise_rate,
            "corrupted_examples": float(self.corrupted_examples),
            "replay_examples_seen": float(self.replay_examples),
            "realized_corruption_rate": (
                self.corrupted_examples / self.replay_examples
                if self.replay_examples
                else 0.0
            ),
        }
