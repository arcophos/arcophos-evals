"""Inspect entry point: importing this module registers every benchmark task.

Declared in pyproject.toml as the ``inspect_ai`` entry point, which is how
``inspect eval arcophos_evals/<task>`` resolves task names from an installed
package. Keeping registration here (rather than in the package root) lets
``arcophos_evals.types`` and ``arcophos_evals.dataset`` import without
inspect_ai installed.
"""
from arcophos_evals.benchmarks import (  # noqa: F401
    hob_blood_pressure,
    hob_cancer_screening,
    hob_exercise_fitness,
    hob_geroscience,
    hob_hormone_optimization,
    hob_incretin_clinical,
    hob_incretin_evidence,
    hob_lipids_ascvd,
    hob_nutrition_supplements,
    hob_sample,
    hob_sleep,
)
