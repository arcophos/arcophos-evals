"""Benchmark registry.

Each Arcophos benchmark in the HealthBench Professional compatible format is
one module here exposing Inspect ``@task`` definitions. Adding a benchmark:
create a module with its task functions, import
them here and in ``_registry`` (the ``inspect_ai`` entry point, which is what
makes ``inspect eval arcophos_evals/<task>`` resolve), and add the names to
``_TASKS`` in the package root for lazy attribute access.
"""
from arcophos_evals.benchmarks.hob import (  # noqa: F401
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
