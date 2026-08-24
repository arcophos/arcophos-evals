"""Inspect ``@task`` definitions for Health Optimization Bench.

Every task is closed book: the solver is a bare ``generate()`` call with no
tools, and grading is delegated to :func:`arcophos_evals.scorer.rubric_scorer`. The
public sample contains 30 tasks, 3 in each of 10 micro benches; the
per-micro-bench tasks filter by the exact display names used in the dataset.

All tasks take the same two parameters: ``source`` (path to a local
HBP-compatible JSONL file; ``None`` uses the canonical public sample) and
``judge`` (passed through to ``rubric_scorer``).
"""
from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.model import Model
from inspect_ai.solver import generate

from arcophos_evals.dataset import filter_micro_bench, load_tasks, to_inspect_dataset
from arcophos_evals.scorer import rubric_scorer

TASK_VERSION = 1
"""Bumped whenever a change could affect scores (tasks, prompts, formula)."""

MICRO_BENCHES = {
    "hob_blood_pressure": "Blood Pressure Optimization",
    "hob_cancer_screening": "Cancer Screening & Early Detection",
    "hob_exercise_fitness": "Exercise & Cardiorespiratory Fitness",
    "hob_geroscience": "Longevity / Geroscience Pharmacology",
    "hob_hormone_optimization": "Hormone Optimization",
    "hob_incretin_clinical": "Incretin Therapeutics: Clinical Decisions",
    "hob_incretin_evidence": "Incretin Therapeutics: Evidence Synthesis",
    "hob_lipids_ascvd": "Lipids & ASCVD Prevention",
    "hob_nutrition_supplements": "Nutrition & Supplements",
    "hob_sleep": "Sleep Optimization",
}
"""Task name to dataset display name; the single source for filter strings."""


def _normalize_judge(
    judge: str | Model | list[str | Model] | None,
) -> str | Model | list[str | Model] | None:
    """Split a comma-joined judge string into the panel list the scorer expects.

    `-T judge="a,b,c"` arrives from the Inspect CLI as one string; anything
    else passes through unchanged.
    """
    if isinstance(judge, str) and "," in judge:
        return [j.strip() for j in judge.split(",") if j.strip()]
    return judge


def _rubric_task(
    source: str | None,
    judge: str | Model | list[str | Model] | None,
    micro_bench: str | None = None,
    cache: bool = True,
) -> Task:
    """Build a closed-book rubric-graded task, optionally for one micro bench."""
    judge = _normalize_judge(judge)
    tasks = load_tasks(source)
    if micro_bench is not None:
        selected = filter_micro_bench(tasks, micro_bench)
        if not selected:
            available = sorted({t.micro_bench for t in tasks})
            raise ValueError(
                f"no tasks with micro_bench {micro_bench!r} in the dataset; "
                f"available micro benches: {available}"
            )
        tasks = selected
    return Task(
        dataset=to_inspect_dataset(tasks),
        solver=generate(cache=cache),
        scorer=rubric_scorer(judge=judge),
        version=TASK_VERSION,
    )


@task
def hob_sample(
    source: str | None = None,
    judge: str | Model | list[str | Model] | None = None,
    cache: bool = True,
) -> Task:
    """Every micro bench: all 30 tasks in the public sample.

    Args:
        source: Path to a local HBP-compatible JSONL file; ``None`` uses the
            canonical public sample.
        judge: Judge model, panel list, or ``None`` for the pinned default
            (see :func:`arcophos_evals.scorer.rubric_scorer`).
        cache: Reuse cached candidate answers for identical (model, task)
            requests across runs (inspect_ai's generation cache). Pass
            ``False`` to resample, e.g. when measuring run-to-run variance.
    """
    return _rubric_task(source, judge, cache=cache)


@task
def hob_blood_pressure(
    source: str | None = None,
    judge: str | Model | list[str | Model] | None = None,
    cache: bool = True,
) -> Task:
    """Blood Pressure Optimization micro bench: 3 of the 30 sample tasks.

    Args:
        source: Path to a local HBP-compatible JSONL file; ``None`` uses the
            canonical public sample.
        judge: Judge model, panel list, or ``None`` for the pinned default
            (see :func:`arcophos_evals.scorer.rubric_scorer`).
        cache: Reuse cached candidate answers for identical (model, task)
            requests across runs (inspect_ai's generation cache). Pass
            ``False`` to resample, e.g. when measuring run-to-run variance.
    """
    return _rubric_task(source, judge, MICRO_BENCHES["hob_blood_pressure"], cache=cache)


@task
def hob_cancer_screening(
    source: str | None = None,
    judge: str | Model | list[str | Model] | None = None,
    cache: bool = True,
) -> Task:
    """Cancer Screening & Early Detection micro bench: 3 of the 30 sample tasks.

    Args:
        source: Path to a local HBP-compatible JSONL file; ``None`` uses the
            canonical public sample.
        judge: Judge model, panel list, or ``None`` for the pinned default
            (see :func:`arcophos_evals.scorer.rubric_scorer`).
        cache: Reuse cached candidate answers for identical (model, task)
            requests across runs (inspect_ai's generation cache). Pass
            ``False`` to resample, e.g. when measuring run-to-run variance.
    """
    return _rubric_task(source, judge, MICRO_BENCHES["hob_cancer_screening"], cache=cache)


@task
def hob_exercise_fitness(
    source: str | None = None,
    judge: str | Model | list[str | Model] | None = None,
    cache: bool = True,
) -> Task:
    """Exercise & Cardiorespiratory Fitness micro bench: 3 of the 30 sample tasks.

    Args:
        source: Path to a local HBP-compatible JSONL file; ``None`` uses the
            canonical public sample.
        judge: Judge model, panel list, or ``None`` for the pinned default
            (see :func:`arcophos_evals.scorer.rubric_scorer`).
        cache: Reuse cached candidate answers for identical (model, task)
            requests across runs (inspect_ai's generation cache). Pass
            ``False`` to resample, e.g. when measuring run-to-run variance.
    """
    return _rubric_task(source, judge, MICRO_BENCHES["hob_exercise_fitness"], cache=cache)


@task
def hob_geroscience(
    source: str | None = None,
    judge: str | Model | list[str | Model] | None = None,
    cache: bool = True,
) -> Task:
    """Longevity / Geroscience Pharmacology micro bench: 3 of the 30 sample tasks.

    Args:
        source: Path to a local HBP-compatible JSONL file; ``None`` uses the
            canonical public sample.
        judge: Judge model, panel list, or ``None`` for the pinned default
            (see :func:`arcophos_evals.scorer.rubric_scorer`).
        cache: Reuse cached candidate answers for identical (model, task)
            requests across runs (inspect_ai's generation cache). Pass
            ``False`` to resample, e.g. when measuring run-to-run variance.
    """
    return _rubric_task(source, judge, MICRO_BENCHES["hob_geroscience"], cache=cache)


@task
def hob_hormone_optimization(
    source: str | None = None,
    judge: str | Model | list[str | Model] | None = None,
    cache: bool = True,
) -> Task:
    """Hormone Optimization micro bench: 3 of the 30 sample tasks.

    Args:
        source: Path to a local HBP-compatible JSONL file; ``None`` uses the
            canonical public sample.
        judge: Judge model, panel list, or ``None`` for the pinned default
            (see :func:`arcophos_evals.scorer.rubric_scorer`).
        cache: Reuse cached candidate answers for identical (model, task)
            requests across runs (inspect_ai's generation cache). Pass
            ``False`` to resample, e.g. when measuring run-to-run variance.
    """
    return _rubric_task(source, judge, MICRO_BENCHES["hob_hormone_optimization"], cache=cache)


@task
def hob_lipids_ascvd(
    source: str | None = None,
    judge: str | Model | list[str | Model] | None = None,
    cache: bool = True,
) -> Task:
    """Lipids & ASCVD Prevention micro bench: 3 of the 30 sample tasks.

    Args:
        source: Path to a local HBP-compatible JSONL file; ``None`` uses the
            canonical public sample.
        judge: Judge model, panel list, or ``None`` for the pinned default
            (see :func:`arcophos_evals.scorer.rubric_scorer`).
        cache: Reuse cached candidate answers for identical (model, task)
            requests across runs (inspect_ai's generation cache). Pass
            ``False`` to resample, e.g. when measuring run-to-run variance.
    """
    return _rubric_task(source, judge, MICRO_BENCHES["hob_lipids_ascvd"], cache=cache)


@task
def hob_nutrition_supplements(
    source: str | None = None,
    judge: str | Model | list[str | Model] | None = None,
    cache: bool = True,
) -> Task:
    """Nutrition & Supplements micro bench: 3 of the 30 sample tasks.

    Args:
        source: Path to a local HBP-compatible JSONL file; ``None`` uses the
            canonical public sample.
        judge: Judge model, panel list, or ``None`` for the pinned default
            (see :func:`arcophos_evals.scorer.rubric_scorer`).
        cache: Reuse cached candidate answers for identical (model, task)
            requests across runs (inspect_ai's generation cache). Pass
            ``False`` to resample, e.g. when measuring run-to-run variance.
    """
    return _rubric_task(source, judge, MICRO_BENCHES["hob_nutrition_supplements"], cache=cache)


@task
def hob_sleep(
    source: str | None = None,
    judge: str | Model | list[str | Model] | None = None,
    cache: bool = True,
) -> Task:
    """Sleep Optimization micro bench: 3 of the 30 sample tasks.

    Args:
        source: Path to a local HBP-compatible JSONL file; ``None`` uses the
            canonical public sample.
        judge: Judge model, panel list, or ``None`` for the pinned default
            (see :func:`arcophos_evals.scorer.rubric_scorer`).
        cache: Reuse cached candidate answers for identical (model, task)
            requests across runs (inspect_ai's generation cache). Pass
            ``False`` to resample, e.g. when measuring run-to-run variance.
    """
    return _rubric_task(source, judge, MICRO_BENCHES["hob_sleep"], cache=cache)


@task
def hob_incretin_evidence(
    source: str | None = None,
    judge: str | Model | list[str | Model] | None = None,
    cache: bool = True,
) -> Task:
    """Incretin Therapeutics: Evidence Synthesis micro bench: 3 of the 30 sample tasks.

    Args:
        source: Path to a local HBP-compatible JSONL file; ``None`` uses the
            canonical public sample.
        judge: Judge model, panel list, or ``None`` for the pinned default
            (see :func:`arcophos_evals.scorer.rubric_scorer`).
        cache: Reuse cached candidate answers for identical (model, task)
            requests across runs (inspect_ai's generation cache). Pass
            ``False`` to resample, e.g. when measuring run-to-run variance.
    """
    return _rubric_task(source, judge, MICRO_BENCHES["hob_incretin_evidence"], cache=cache)


@task
def hob_incretin_clinical(
    source: str | None = None,
    judge: str | Model | list[str | Model] | None = None,
    cache: bool = True,
) -> Task:
    """Incretin Therapeutics: Clinical Decisions micro bench: 3 of the 30 sample tasks.

    Args:
        source: Path to a local HBP-compatible JSONL file; ``None`` uses the
            canonical public sample.
        judge: Judge model, panel list, or ``None`` for the pinned default
            (see :func:`arcophos_evals.scorer.rubric_scorer`).
        cache: Reuse cached candidate answers for identical (model, task)
            requests across runs (inspect_ai's generation cache). Pass
            ``False`` to resample, e.g. when measuring run-to-run variance.
    """
    return _rubric_task(source, judge, MICRO_BENCHES["hob_incretin_clinical"], cache=cache)
