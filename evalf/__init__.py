"""evalf public package exports."""

from importlib.metadata import PackageNotFoundError, version

from .evaluation import Evaluator, a_evaluate, evaluate
from .schemas import EvalCase, MetricResult, RunReport, SampleResult

__all__ = [
    "__version__",
    "Evaluator",
    "EvalCase",
    "MetricResult",
    "RunReport",
    "SampleResult",
    "a_evaluate",
    "evaluate",
]

try:
    __version__ = version("evalf")
except PackageNotFoundError:
    __version__ = "0+unknown"
