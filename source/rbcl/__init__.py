"""Research code for Prior-Guided Risk-Budgeted Continual Learning."""

from .budgeting import RiskBudgetingPlugin
from .data import apply_deferred_sample_clock, apply_sample_clock, build_benchmark
from .evaluation import build_evaluator
from .models import build_model
from .retention import ConsequenceAwareExperienceBalancedBuffer
from .trainer import build_strategy, run_stream

__all__ = [
    "RiskBudgetingPlugin",
    "build_benchmark",
    "apply_sample_clock",
    "apply_deferred_sample_clock",
    "build_evaluator",
    "build_model",
    "ConsequenceAwareExperienceBalancedBuffer",
    "build_strategy",
    "run_stream",
]
