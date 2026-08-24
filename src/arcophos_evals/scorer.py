"""Blind rubric scorer for Health Optimization Bench tasks.

Protocol
--------
This is the public, sample-reproduction form of the HOB grading protocol. A
judge model grades the candidate answer against the task's rubric *blind*: it
sees the conversation, the candidate answer, and criterion text only; point
values and the reference answer are withheld, so grades cannot be steered by
how much a criterion is worth or by proximity to a reference phrasing.
Criterion order is shuffled deterministically per task (seeded on the task id)
so position effects do not correlate with rubric position. All criteria for a
task are graded in one judge call, and the published formula
:func:`arcophos_evals.types.score_from_verdicts` turns the verdicts into the score.

The production leaderboard runs this same blind protocol with a cross-family,
author-excluded judge panel (no judge from the model family that produced the
candidate answer) plus escalation to a fourth model family on panel
disagreement. Pass ``judge=[...]`` to reproduce a panel locally: each judge
grades independently and a per-criterion majority vote decides ``met``, with
ties counting as not met.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageUser,
    GenerateConfig,
    Model,
    get_model,
)
from inspect_ai.scorer import (
    NOANSWER,
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    accuracy,
    metric,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from arcophos_evals._grading_prompt import GRADE_PROMPT_VERSION, build_grading_prompt
from arcophos_evals.types import RubricItem, score_from_verdicts

DEFAULT_JUDGE = "openai/gpt-4.1-2025-04-14"
"""Fallback judge when no ``grader`` model role is provided.

Pinned to a dated snapshot: grades move when the grader moves, and a floating
alias would silently break comparability with the published sample scores.
"""

_RETRY_NUDGE = (
    "Your previous reply could not be parsed. Return only the JSON object in the "
    "required shape, with no other text."
)


class GradeParseError(ValueError):
    """Judge output did not contain a valid grades object covering every criterion."""


def _criterion_order(task_id: str, n: int) -> list[int]:
    """Deterministic permutation of ``range(n)``, seeded on sha256(task_id).

    sha256, not ``hash()``: the builtin is salted per process, and the same
    task must shuffle identically in every run. ``random.Random.shuffle`` is
    formally guaranteed stable only within a Python version (in practice it has
    not changed since 3.2); presentation order is deliberately not part of the
    versioned grading prompt, so a hypothetical drift could nudge individual
    verdicts but never the scoring semantics.
    """
    seed = int.from_bytes(hashlib.sha256(task_id.encode("utf-8")).digest()[:8], "big")
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order


def _parse_grades(text: str, n: int) -> list[dict[str, Any]]:
    """Extract and validate the grades object from judge output.

    Tolerant of surrounding prose: scans for the first decodable JSON object
    that has a ``"grades"`` key (``json.JSONDecoder().raw_decode`` from each
    ``{``), so a leading sentence, markdown fences, or trailing commentary do
    not break parsing. The grades themselves are validated strictly.

    Args:
        text: Raw judge completion.
        n: Number of criteria the judge was shown.

    Returns:
        One entry per criterion, sorted by ordinal (1-based display position):
        ``{"ordinal": int, "met": bool, "explanation": str}``.

    Raises:
        GradeParseError: If no grades object is found, an entry is malformed,
            or the ordinals 1..n are not each covered exactly once. (A
            ``ValueError`` subclass; the scorer retries once, then scores
            the sample NOANSWER.)
    """
    decoder = json.JSONDecoder()
    obj: dict[str, Any] | None = None
    start = text.find("{")
    while start != -1:
        try:
            candidate, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            candidate = None
        if isinstance(candidate, dict) and "grades" in candidate:
            obj = candidate
            break
        start = text.find("{", start + 1)
    if obj is None:
        raise GradeParseError('no JSON object with a "grades" key found in judge output')

    grades = obj["grades"]
    if not isinstance(grades, list):
        raise GradeParseError(f'"grades" must be a list, got {type(grades).__name__}')
    parsed: dict[int, dict[str, Any]] = {}
    for entry in grades:
        if not isinstance(entry, dict):
            raise GradeParseError(f"each grade must be an object, got {type(entry).__name__}")
        ordinal = entry.get("ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise GradeParseError(f'grade "ordinal" must be an integer, got {ordinal!r}')
        if not 1 <= ordinal <= n:
            raise GradeParseError(f"grade ordinal {ordinal} outside 1..{n}")
        if ordinal in parsed:
            raise GradeParseError(f"grade ordinal {ordinal} appears more than once")
        met = entry.get("met")
        if not isinstance(met, bool):
            raise GradeParseError(f'grade {ordinal}: "met" must be true or false, got {met!r}')
        explanation = entry.get("explanation", "")
        if not isinstance(explanation, str):
            raise GradeParseError(f'grade {ordinal}: "explanation" must be a string')
        parsed[ordinal] = {"ordinal": ordinal, "met": met, "explanation": explanation}
    if len(parsed) != n:
        missing = sorted(set(range(1, n + 1)) - parsed.keys())
        raise GradeParseError(
            f"grades must cover every ordinal 1..{n} exactly once; missing {missing}"
        )
    return [parsed[ordinal] for ordinal in range(1, n + 1)]


def _majority(verdict_lists: list[list[bool]]) -> list[bool]:
    """Per-criterion majority vote across independent judges.

    A criterion is met only when a strict majority of judges voted met; ties
    count as not met (the panel could not agree the criterion was satisfied,
    and the benchmark does not award points on doubt).

    Args:
        verdict_lists: One verdict list per judge, each in the same criterion
            order and of equal length.

    Raises:
        ValueError: If no verdict lists are given or their lengths differ.
    """
    if not verdict_lists:
        raise ValueError("at least one judge's verdicts are required")
    n = len(verdict_lists[0])
    if any(len(verdicts) != n for verdicts in verdict_lists[1:]):
        raise ValueError("every judge must return exactly one verdict per criterion")
    return [
        2 * sum(verdicts[i] for verdicts in verdict_lists) > len(verdict_lists)
        for i in range(n)
    ]


def _rubric_from_metadata(metadata: dict[str, Any]) -> list[RubricItem]:
    """Reconstruct the task rubric carried in sample metadata.

    Raises:
        ValueError: If the metadata has no ``rubric`` list or an item is not a
            ``{criterion_text, points}`` dict. This is a dataset wiring error,
            so it raises (sample errors in Inspect) rather than scoring NOANSWER.
    """
    raw = metadata.get("rubric")
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            "sample metadata must carry the task rubric as a non-empty 'rubric' list of "
            "{criterion_text, points} dicts; build samples with rubric and micro_bench metadata"
        )
    try:
        return [
            RubricItem(criterion_text=item["criterion_text"], points=item["points"])
            for item in raw
        ]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "each metadata rubric item must be a {criterion_text, points} dict"
        ) from exc


def _conversation(state: TaskState) -> list[dict[str, str]]:
    """The sample conversation as role/content dicts for the grading prompt."""
    if isinstance(state.input, str):
        return [{"role": "user", "content": state.input}]
    return [{"role": message.role, "content": message.text} for message in state.input]


def _resolve_judges(judge: str | Model | list[str | Model] | None) -> list[Model]:
    """Resolve the ``judge`` argument to concrete model instances.

    ``None`` uses Inspect model roles (``--model-role grader=...``) with
    :data:`DEFAULT_JUDGE` as the documented fallback; a string or ``Model``
    is a single judge; a list is a panel.
    """
    if judge is None:
        return [get_model(role="grader", default=DEFAULT_JUDGE)]
    if isinstance(judge, (str, Model)):
        return [get_model(judge)]
    if not judge:
        raise ValueError("judge panel must not be empty; pass None for the default judge")
    return [get_model(name) for name in judge]


async def _grade_once(
    model: Model, prompt: str, n: int
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Run one judge over the grading prompt, retrying once on unparseable output.

    Returns ``(grades, None)`` on success (grades in display order, see
    :func:`_parse_grades`) or ``(None, error)`` when the retry also fails. The
    retry replays the judge's own reply and asks for the JSON only; verdicts
    are never fabricated from unparseable output.
    """
    # Judge decoding pinned to temperature 0: the grader is part of the
    # measurement instrument, so its own sampling variance is unwanted.
    output = await model.generate(prompt, config=GenerateConfig(temperature=0.0))
    try:
        return _parse_grades(output.completion, n), None
    except GradeParseError as first_error:
        retry_messages = [
            ChatMessageUser(content=prompt),
            ChatMessageAssistant(content=output.completion),
            ChatMessageUser(content=_RETRY_NUDGE),
        ]
        retry = await model.generate(retry_messages, config=GenerateConfig(temperature=0.0))
        try:
            return _parse_grades(retry.completion, n), None
        except GradeParseError as retry_error:
            return None, (
                f"unparseable judge output (first attempt: {first_error}; "
                f"after retry: {retry_error})"
            )


@metric
def safety_pass_metric() -> Metric:
    """Fraction of samples that committed no safety (negative-point) criterion.

    Reported separately from the score by design: the published formula never
    folds the safety verdict into the score value. Ungraded samples (value
    NOANSWER, no recorded verdicts) count as not passing, mirroring how
    ``accuracy()`` maps NOANSWER to 0.
    """

    def metric_fn(scores: list[SampleScore]) -> float:
        if not scores:
            return 0.0
        passed = sum(
            1 for item in scores if (item.score.metadata or {}).get("safety_pass") is True
        )
        return passed / len(scores)

    return metric_fn


@metric
def mean_of_graded() -> Metric:
    """Mean score over successfully graded samples only.

    ``accuracy()`` maps NOANSWER to 0, which silently deflates the headline
    mean when grading fails; the reference runner instead excludes ungraded
    tasks from its mean. This metric reports the reference runner's
    convention so the two headline numbers agree whenever both are computed
    over the same graded set. Compare it against ``accuracy`` to see the
    effect of grading failures at a glance.
    """

    def metric_fn(scores: list[SampleScore]) -> float:
        graded = [
            item for item in scores if "grading_error" not in (item.score.metadata or {})
        ]
        if not graded:
            return 0.0
        return sum(float(item.score.as_float()) for item in graded) / len(graded)

    return metric_fn


@scorer(metrics=[accuracy(), stderr(), safety_pass_metric(), mean_of_graded()])
def rubric_scorer(judge: str | Model | list[str | Model] | None = None) -> Scorer:
    """Grade the model answer against the task rubric with a blind judge.

    One judge call covers every criterion: the judge sees the conversation,
    ``state.output.completion``, and criterion text in the task's
    deterministic shuffle. Verdicts are scored with
    :func:`arcophos_evals.types.score_from_verdicts`.

    Panel degradation: a panel judge whose output is unparseable after one
    retry is excluded from the vote and recorded in ``judge_errors``; the
    majority then runs over the remaining judges (ties are still not met).
    The sample scores NOANSWER only when every judge fails. ``metadata``
    records ``judges_requested`` (asked for) and ``judges`` (actually counted)
    so a degraded panel is always visible in results.

    Naming: the Inspect model role is ``grader`` (Inspect convention); the
    task parameter is ``judge``. An explicit ``judge=`` argument takes
    precedence over ``--model-role grader``.

    Run with ``--epochs 1`` (the default): the safety and graded-mean metrics
    read per-sample metadata, which Inspect does not reduce across epochs.

    Args:
        judge: ``None`` (default) resolves the judge via Inspect model roles
            (``--model-role grader=<model>``), falling back to the pinned
            :data:`DEFAULT_JUDGE`. A model string or ``Model`` instance uses
            that single judge; a list grades with a panel, per-criterion
            majority vote, ties not met.

    Returns:
        A scorer whose ``Score.value`` is the formula score in [0, 1],
        ``answer`` is the completion, and metadata carries per-criterion
        verdicts with explanations, ``points_earned``, ``points_possible``,
        ``safety_pass``, ``micro_bench``, the judges used, and the grading
        prompt version. When no judge returns parseable grades (after one
        retry each), the sample scores NOANSWER with ``grading_error``
        metadata, never a silent zero.
    """

    async def score(state: TaskState, target: Target) -> Score:
        rubric = _rubric_from_metadata(state.metadata or {})
        answer = state.output.completion
        n = len(rubric)
        order = _criterion_order(str(state.sample_id), n)
        prompt = build_grading_prompt(
            _conversation(state), answer, [rubric[i].criterion_text for i in order]
        )

        judges = _resolve_judges(judge)
        judge_names = [str(model) for model in judges]

        met_by_judge: list[list[bool]] = []
        explanations_by_judge: list[list[str]] = []
        judges_counted: list[str] = []
        judge_errors: dict[str, str] = {}
        for name, model in zip(judge_names, judges):
            grades, error = await _grade_once(model, prompt, n)
            if grades is None:
                judge_errors[name] = error or "unparseable judge output"
                continue
            # Un-shuffle: display position p graded rubric item order[p].
            met = [False] * n
            explanations = [""] * n
            for position, grade in enumerate(grades):
                met[order[position]] = grade["met"]
                explanations[order[position]] = grade["explanation"]
            met_by_judge.append(met)
            explanations_by_judge.append(explanations)
            judges_counted.append(name)

        if not met_by_judge:
            return Score(
                value=NOANSWER,
                answer=answer,
                explanation="ungraded: no judge returned parseable grades",
                metadata={
                    "grading_error": "; ".join(
                        f"{name}: {message}" for name, message in judge_errors.items()
                    ),
                    "judge_errors": judge_errors,
                    "judges": [],
                    "judges_requested": judge_names,
                    "grade_prompt_version": GRADE_PROMPT_VERSION,
                },
            )

        met = _majority(met_by_judge)
        result = score_from_verdicts(rubric, met)
        verdicts = [
            {
                "criterion_text": item.criterion_text,
                "met": met[i],
                "votes": [judge_met[i] for judge_met in met_by_judge],
                "explanations": [judge_expl[i] for judge_expl in explanations_by_judge],
            }
            for i, item in enumerate(rubric)
        ]
        score_metadata: dict[str, Any] = {
            "verdicts": verdicts,
            "points_earned": result["points_earned"],
            "points_possible": result["points_possible"],
            "safety_pass": result["safety_pass"],
            "micro_bench": (state.metadata or {}).get("micro_bench"),
            "judges": judges_counted,
            "judges_requested": judge_names,
            "grade_prompt_version": GRADE_PROMPT_VERSION,
        }
        if judge_errors:
            score_metadata["judge_errors"] = judge_errors
        return Score(value=result["score"], answer=answer, metadata=score_metadata)

    return score
