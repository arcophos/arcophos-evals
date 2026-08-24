"""End-to-end tests of the rubric scorer through a real Inspect eval.

Unlike tests/test_scorer.py (pure helpers, no model calls), these run
``inspect_ai.eval`` over the fixture tasks with mockllm models: a canned
candidate and a judge scripted via ``custom_outputs`` to return exact grade
JSON. That exercises the full path the leaderboard uses (dataset loading,
solver, grading prompt, judge call, retry, verdict un-shuffling, formula,
Score metadata) with deterministic verdicts.

The scripted judge is wired in through ``judge=None`` plus
``model_roles={"grader": ...}`` on the eval call, exercising the documented
``--model-role grader=`` resolution path. (``judge=`` also accepts ``Model``
instances directly.)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("inspect_ai")

from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import NOANSWER

from arcophos_evals._grading_prompt import GRADE_PROMPT_VERSION
from arcophos_evals.benchmarks.hob import hob_sample
from arcophos_evals.dataset import load_tasks
from arcophos_evals.scorer import _criterion_order
from arcophos_evals.types import HOBTask

FIXTURES = str(Path(__file__).resolve().parent / "fixtures.jsonl")

TOLERANCE = 1e-9


def _scripted_model(completions: list[str]):
    """A mockllm model that returns the given completions in order."""
    return get_model(
        "mockllm/model",
        memoize=False,
        custom_outputs=[
            ModelOutput.from_content(model="mockllm/model", content=completion)
            for completion in completions
        ],
    )


def _grades_json(task: HOBTask, desired_met: list[bool]) -> str:
    """Valid judge JSON that yields ``desired_met`` (in rubric order) for a task.

    The scorer shows the judge criteria in a per-task deterministic shuffle and
    maps display ordinal p back to rubric index ``order[p - 1]``, so the
    scripted verdict for display ordinal p must be the desired verdict of the
    rubric item shown at that position.
    """
    n = len(task.rubric)
    assert len(desired_met) == n
    order = _criterion_order(task.task_id, n)
    grades = [
        {"ordinal": p + 1, "met": bool(desired_met[order[p]]), "explanation": "scripted"}
        for p in range(n)
    ]
    return json.dumps({"grades": grades})


def _run_eval(judge, tmp_path, judge_override=None, **kwargs):
    """One in-process Inspect eval over the fixture tasks with a scripted judge.

    ``max_samples=1`` forces serial sample execution so the judge's
    ``custom_outputs`` are consumed in dataset order. ``judge_override`` passes
    a judge panel (list of Model instances) directly through the task's
    ``judge=`` parameter instead of the grader model role.
    """
    logs = inspect_eval(
        hob_sample(source=FIXTURES, judge=judge_override),
        model=get_model("mockllm/model", memoize=False),  # canned candidate answer
        model_roles={"grader": judge},
        log_dir=str(tmp_path),
        max_samples=1,
        display="none",
        **kwargs,
    )
    assert len(logs) == 1
    assert logs[0].status == "success"
    return logs[0]


def test_end_to_end_scores_match_published_formula(tmp_path):
    tasks = load_tasks(FIXTURES)
    assert len(tasks) == 2
    first, second = tasks
    for task in tasks:  # exactly one safety criterion each, or the cases below are meaningless
        assert sum(1 for item in task.rubric if item.points < 0) == 1

    # Sample 1: every positive criterion met, the safety criterion NOT met.
    first_met = [item.points > 0 for item in first.rubric]
    # Sample 2: everything met, including the safety criterion (subtraction case).
    second_met = [True] * len(second.rubric)
    judge = _scripted_model(
        [_grades_json(first, first_met), _grades_json(second, second_met)]
    )

    log = _run_eval(judge, tmp_path)
    samples = {sample.id: sample for sample in log.samples}
    assert set(samples) == {first.task_id, second.task_id}

    # Sample 1: all positives earned, safety clean -> exactly 1.0
    first_possible = sum(item.points for item in first.rubric if item.points > 0)
    score = samples[first.task_id].scores["rubric_scorer"]
    assert score.value == pytest.approx(1.0, abs=TOLERANCE)
    metadata = score.metadata or {}
    assert metadata["points_earned"] == first_possible
    assert metadata["points_possible"] == first_possible
    assert metadata["safety_pass"] is True
    assert metadata["grade_prompt_version"] == GRADE_PROMPT_VERSION
    assert "judges" in metadata  # judges_requested may also exist; assert presence only
    verdicts = metadata["verdicts"]
    assert [verdict["met"] for verdict in verdicts] == first_met
    assert [verdict["criterion_text"] for verdict in verdicts] == [
        item.criterion_text for item in first.rubric
    ]

    # Sample 2: committed safety criterion subtracts from the score
    second_possible = sum(item.points for item in second.rubric if item.points > 0)
    second_earned = sum(item.points for item in second.rubric)  # all met, negative included
    expected_second = max(0.0, min(1.0, second_earned / second_possible))
    assert 0.0 < expected_second < 1.0  # the fixture makes the subtraction observable
    score = samples[second.task_id].scores["rubric_scorer"]
    assert score.value == pytest.approx(expected_second, abs=TOLERANCE)
    metadata = score.metadata or {}
    assert metadata["points_earned"] == second_earned
    assert metadata["points_possible"] == second_possible
    assert metadata["safety_pass"] is False
    assert metadata["grade_prompt_version"] == GRADE_PROMPT_VERSION
    assert "judges" in metadata
    assert [verdict["met"] for verdict in metadata["verdicts"]] == second_met


def test_unparseable_judge_output_scores_noanswer(tmp_path):
    # Both the first attempt and the one retry return junk -> NOANSWER, never 0.
    judge = _scripted_model(
        ["I decline to emit JSON.", "still no grades object here: {oops"]
    )
    log = _run_eval(judge, tmp_path, limit=1)  # one sample = exactly two judge calls
    assert len(log.samples) == 1
    score = log.samples[0].scores["rubric_scorer"]
    assert score.value == NOANSWER
    metadata = score.metadata or {}
    assert metadata.get("grading_error")
    assert "unparseable" in metadata["grading_error"]
    assert "judges" in metadata
    assert metadata["grade_prompt_version"] == GRADE_PROMPT_VERSION


def test_garbled_judge_recovers_via_retry_nudge(tmp_path):
    # First reply junk, nudge retry valid -> a normal score, no grading_error.
    tasks = load_tasks(FIXTURES)
    first = tasks[0]
    all_met = [True] * len(first.rubric)
    judge = _scripted_model(["I refuse to emit JSON.", _grades_json(first, all_met)])
    log = _run_eval(judge, tmp_path, limit=1)
    score = log.samples[0].scores["rubric_scorer"]
    assert score.value != NOANSWER
    metadata = score.metadata or {}
    assert not metadata.get("grading_error")
    assert [verdict["met"] for verdict in metadata["verdicts"]] == all_met


def test_panel_degrades_gracefully_when_one_judge_fails(tmp_path):
    # A three-judge panel where one judge is unparseable on both attempts:
    # the vote proceeds over the two survivors and the failure is recorded.
    tasks = load_tasks(FIXTURES)
    first = tasks[0]
    desired = [item.points > 0 for item in first.rubric]
    good = _grades_json(first, desired)
    panel = [
        _scripted_model([good]),
        _scripted_model(["no json here", "still no json"]),  # attempt + nudge
        _scripted_model([good]),
    ]
    log = _run_eval(panel[0], tmp_path, limit=1, judge_override=panel)
    score = log.samples[0].scores["rubric_scorer"]
    assert score.value != NOANSWER
    assert score.value == pytest.approx(1.0, abs=TOLERANCE)
    metadata = score.metadata or {}
    assert len(metadata["judges_requested"]) == 3
    assert len(metadata["judges"]) == 2
    assert len(metadata["judge_errors"]) == 1
    assert "unparseable" in next(iter(metadata["judge_errors"].values()))


def test_judge_comma_string_normalizes_to_panel():
    from arcophos_evals.benchmarks.hob import _normalize_judge

    assert _normalize_judge("a,b , c,") == ["a", "b", "c"]
    assert _normalize_judge("provider/model") == "provider/model"
    assert _normalize_judge(None) is None
