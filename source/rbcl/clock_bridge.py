"""Recent-window bridge replay for task-ID-free sample-clock scheduling."""

from __future__ import annotations

from typing import Dict

from avalanche.benchmarks.utils.utils import concat_datasets
from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.storage_policy import ReservoirSamplingBuffer


class ClockBridgeReplayPlugin(SupervisedPlugin):
    """Replay a fixed mix of global reservoir and recent clock-window tail.

    This plugin is used only after the stream has been transformed into fixed
    sample-clock windows. The recent tail supplies a task-ID-free local bridge
    across adjacent windows, while the reservoir preserves long-term memory.
    """

    def __init__(self, *, mem_size: int, bridge_ratio: float = 0.5) -> None:
        super().__init__()
        if mem_size <= 1:
            raise ValueError("mem_size must be greater than one")
        if not 0.0 < bridge_ratio < 1.0:
            raise ValueError("bridge_ratio must be in (0, 1)")
        self.mem_size = int(mem_size)
        self.bridge_size = max(1, int(round(mem_size * bridge_ratio)))
        self.reservoir_size = mem_size - self.bridge_size
        self.reservoir = ReservoirSamplingBuffer(max_size=self.reservoir_size)
        self.recent = concat_datasets([])
        self._clock_updates = 0

    @property
    def buffer(self):
        return self.reservoir.buffer.concat(self.recent)

    def before_training_exp(
        self,
        strategy,
        num_workers: int = 0,
        shuffle: bool = True,
        **kwargs,
    ) -> None:
        memory = self.buffer
        if len(memory) == 0:
            return
        # Build the same two-source loader used by Avalanche replay, with the
        # global reservoir and local bridge tail concatenated as its memory.
        from avalanche.benchmarks.utils.data_loader import ReplayDataLoader

        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            memory,
            oversample_small_tasks=True,
            batch_size=strategy.train_mb_size,
            batch_size_mem=strategy.train_mb_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )

    def after_training_exp(self, strategy, **kwargs) -> None:
        dataset = strategy.experience.dataset
        self.reservoir.update_from_dataset(dataset)
        tail_start = max(0, len(dataset) - self.bridge_size)
        self.recent = dataset.subset(range(tail_start, len(dataset)))
        self._clock_updates += 1

    def summary(self) -> Dict[str, object]:
        return {
            "mem_size": self.mem_size,
            "reservoir_size": self.reservoir_size,
            "bridge_size": self.bridge_size,
            "clock_updates": self._clock_updates,
            "uses_task_id": False,
            "recent_buffer_size": len(self.recent),
            "reservoir_buffer_size": len(self.reservoir.buffer),
        }
