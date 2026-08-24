"""Arcophos Evals: Inspect task packs for Arcophos health benchmarks.

The package root exports only the stdlib-safe surface (types, loader helpers).
Benchmark tasks require inspect_ai and load lazily on attribute access, so
``import arcophos_evals`` works in environments without Inspect while
``arcophos_evals.hob_sample`` still resolves where it is installed.
"""
from arcophos_evals.types import HOBTask, RubricItem, score_from_verdicts  # noqa: F401

__version__ = "0.1.0"

_TASKS = (
    "hob_blood_pressure",
    "hob_cancer_screening",
    "hob_exercise_fitness",
    "hob_geroscience",
    "hob_hormone_optimization",
    "hob_incretin_clinical",
    "hob_incretin_evidence",
    "hob_lipids_ascvd",
    "hob_nutrition_supplements",
    "hob_sample",
    "hob_sleep",
)


def __getattr__(name: str):  # PEP 562: lazy task access without importing inspect_ai eagerly
    if name in _TASKS:
        from arcophos_evals import benchmarks

        return getattr(benchmarks, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *_TASKS])
