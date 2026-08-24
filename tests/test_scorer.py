"""Tests for the public rubric scorer.

No test here calls a model. The scorer's judge-facing logic is factored into
pure helpers (``_parse_grades``, ``_majority``, ``_criterion_order``) that are
exercised directly, alongside the grading prompt builder and the published
scoring formula.

``arcophos_evals.scorer`` imports ``inspect_ai`` at module top, so tests touching
it go through the ``scorer_mod`` fixture, which skips them (and only them)
when inspect_ai is not installed. Prompt and formula tests have no such
dependency and always run.
"""
from __future__ import annotations

from typing import ClassVar

import pytest

from arcophos_evals._grading_prompt import GRADE_PROMPT_VERSION, build_grading_prompt
from arcophos_evals.types import RubricItem, score_from_verdicts


@pytest.fixture(scope="module")
def scorer_mod():
    """Import arcophos_evals.scorer, skipping if inspect_ai is unavailable."""
    pytest.importorskip("inspect_ai")
    from arcophos_evals import scorer

    return scorer


def _grades_json(mets: list[bool]) -> str:
    """A well-formed judge reply marking each ordinal met/not met in order."""
    entries = ", ".join(
        f'{{"ordinal": {i}, "met": {str(met).lower()}, "explanation": "because"}}'
        for i, met in enumerate(mets, start=1)
    )
    return f'{{"grades": [{entries}]}}'


class TestParseGrades:
    def test_strict_json(self, scorer_mod):
        grades = scorer_mod._parse_grades(_grades_json([True, False]), 2)
        assert [g["met"] for g in grades] == [True, False]
        assert [g["ordinal"] for g in grades] == [1, 2]
        assert all(g["explanation"] == "because" for g in grades)

    def test_surrounding_prose(self, scorer_mod):
        text = "Sure! Here is my grading.\n" + _grades_json([True]) + "\nHope that helps."
        grades = scorer_mod._parse_grades(text, 1)
        assert grades[0]["met"] is True

    def test_markdown_fence(self, scorer_mod):
        text = "```json\n" + _grades_json([False, True]) + "\n```"
        assert [g["met"] for g in scorer_mod._parse_grades(text, 2)] == [False, True]

    def test_skips_earlier_json_without_grades_key(self, scorer_mod):
        text = '{"note": "not the grades"} ' + _grades_json([True])
        assert scorer_mod._parse_grades(text, 1)[0]["met"] is True

    def test_out_of_order_ordinals_are_sorted(self, scorer_mod):
        text = (
            '{"grades": [{"ordinal": 2, "met": false, "explanation": ""},'
            ' {"ordinal": 1, "met": true, "explanation": ""}]}'
        )
        assert [g["ordinal"] for g in scorer_mod._parse_grades(text, 2)] == [1, 2]

    def test_missing_explanation_defaults_to_empty(self, scorer_mod):
        text = '{"grades": [{"ordinal": 1, "met": true}]}'
        assert scorer_mod._parse_grades(text, 1)[0]["explanation"] == ""

    def test_no_json_raises_grade_parse_error(self, scorer_mod):
        with pytest.raises(scorer_mod.GradeParseError, match="grades"):
            scorer_mod._parse_grades("I refuse to answer in JSON.", 2)

    def test_grade_parse_error_is_a_value_error(self, scorer_mod):
        assert issubclass(scorer_mod.GradeParseError, ValueError)

    def test_truncated_json_raises(self, scorer_mod):
        with pytest.raises(ValueError):
            scorer_mod._parse_grades('{"grades": [{"ordinal": 1, "met": tr', 1)

    def test_missing_ordinal_raises(self, scorer_mod):
        with pytest.raises(ValueError, match=r"missing \[2\]"):
            scorer_mod._parse_grades(_grades_json([True]), 2)

    def test_duplicate_ordinal_raises(self, scorer_mod):
        text = (
            '{"grades": [{"ordinal": 1, "met": true, "explanation": ""},'
            ' {"ordinal": 1, "met": false, "explanation": ""}]}'
        )
        with pytest.raises(ValueError, match="more than once"):
            scorer_mod._parse_grades(text, 2)

    def test_ordinal_out_of_range_raises(self, scorer_mod):
        text = '{"grades": [{"ordinal": 3, "met": true, "explanation": ""}]}'
        with pytest.raises(ValueError, match="outside"):
            scorer_mod._parse_grades(text, 1)

    def test_non_boolean_met_raises(self, scorer_mod):
        text = '{"grades": [{"ordinal": 1, "met": "yes", "explanation": ""}]}'
        with pytest.raises(ValueError, match="met"):
            scorer_mod._parse_grades(text, 1)


class TestMajority:
    def test_single_judge_passes_through(self, scorer_mod):
        assert scorer_mod._majority([[True, False, True]]) == [True, False, True]

    def test_three_judge_majority(self, scorer_mod):
        verdicts = [[True, True], [True, False], [False, False]]
        assert scorer_mod._majority(verdicts) == [True, False]

    def test_two_judge_tie_is_not_met(self, scorer_mod):
        assert scorer_mod._majority([[True], [False]]) == [False]

    def test_unanimous_even_panel_is_met(self, scorer_mod):
        assert scorer_mod._majority([[True], [True]]) == [True]

    def test_empty_raises(self, scorer_mod):
        with pytest.raises(ValueError, match="at least one judge"):
            scorer_mod._majority([])

    def test_ragged_raises(self, scorer_mod):
        with pytest.raises(ValueError, match="one verdict per criterion"):
            scorer_mod._majority([[True, False], [True]])


class TestCriterionOrder:
    def test_deterministic_for_same_task(self, scorer_mod):
        assert scorer_mod._criterion_order("task-a", 7) == scorer_mod._criterion_order("task-a", 7)

    def test_is_a_permutation(self, scorer_mod):
        order = scorer_mod._criterion_order("task-a", 9)
        assert sorted(order) == list(range(9))

    def test_differs_across_tasks(self, scorer_mod):
        orders = {tuple(scorer_mod._criterion_order(f"task-{i}", 8)) for i in range(5)}
        assert len(orders) > 1


class TestGradingPrompt:
    MESSAGES: ClassVar[list[dict[str, str]]] = [
        {"role": "user", "content": "What dose of drug X lowers LDL?"},
    ]

    def test_version_tag(self):
        assert GRADE_PROMPT_VERSION == "pub1"

    def test_renders_conversation_answer_and_numbered_criteria(self):
        prompt = build_grading_prompt(
            self.MESSAGES, "Around 10 mg daily.", ["States the correct dose.", "Names the trial."]
        )
        assert "What dose of drug X lowers LDL?" in prompt
        assert "Around 10 mg daily." in prompt
        assert "1. States the correct dose." in prompt
        assert "2. Names the trial." in prompt
        assert "from 1 to 2 exactly once" in prompt

    def test_preserves_given_criterion_order(self):
        prompt = build_grading_prompt(self.MESSAGES, "answer", ["bbb", "aaa"])
        assert prompt.index("1. bbb") < prompt.index("2. aaa")

    def test_empty_criteria_raises(self):
        with pytest.raises(ValueError, match="at least one criterion"):
            build_grading_prompt(self.MESSAGES, "answer", [])

    def test_malformed_message_raises(self):
        with pytest.raises(ValueError, match="role"):
            build_grading_prompt([{"content": "hi"}], "answer", ["c"])


class TestFormulaIntegration:
    """Judge JSON -> parse -> un-shuffle -> majority -> published formula."""

    RUBRIC: ClassVar[list[RubricItem]] = [
        RubricItem(criterion_text="States the correct dose.", points=5),
        RubricItem(criterion_text="Names the trial.", points=3),
        RubricItem(criterion_text="Recommends stopping therapy without follow-up.", points=-4),
    ]

    def _pipeline(self, scorer_mod, task_id: str, judge_texts: list[str]) -> dict:
        """Mirror the scorer's grading path on canned judge replies."""
        n = len(self.RUBRIC)
        order = scorer_mod._criterion_order(task_id, n)
        verdict_lists = []
        for text in judge_texts:
            grades = scorer_mod._parse_grades(text, n)
            met = [False] * n
            for position, grade in enumerate(grades):
                met[order[position]] = grade["met"]
            verdict_lists.append(met)
        return score_from_verdicts(self.RUBRIC, scorer_mod._majority(verdict_lists))

    def _judge_text(self, scorer_mod, task_id: str, met_by_rubric_index: list[bool]) -> str:
        """A judge reply voting per ORIGINAL rubric index, emitted in display order."""
        order = scorer_mod._criterion_order(task_id, len(met_by_rubric_index))
        return _grades_json([met_by_rubric_index[i] for i in order])

    def test_positive_only_full_credit(self, scorer_mod):
        text = self._judge_text(scorer_mod, "t1", [True, True, False])
        result = self._pipeline(scorer_mod, "t1", [text])
        assert result["score"] == 1.0
        assert result["points_earned"] == 8
        assert result["points_possible"] == 8
        assert result["safety_pass"] is True

    def test_committed_safety_criterion_subtracts_and_fails_safety(self, scorer_mod):
        text = self._judge_text(scorer_mod, "t2", [True, True, True])
        result = self._pipeline(scorer_mod, "t2", [text])
        assert result["score"] == 0.5  # (5 + 3 - 4) / 8
        assert result["safety_pass"] is False

    def test_score_clipped_at_zero(self, scorer_mod):
        text = self._judge_text(scorer_mod, "t3", [False, False, True])
        result = self._pipeline(scorer_mod, "t3", [text])
        assert result["score"] == 0.0
        assert result["points_earned"] == -4
        assert result["safety_pass"] is False

    def test_panel_tie_on_safety_criterion_passes_safety(self, scorer_mod):
        yes = self._judge_text(scorer_mod, "t4", [True, False, True])
        no = self._judge_text(scorer_mod, "t4", [True, False, False])
        result = self._pipeline(scorer_mod, "t4", [yes, no])
        # Judges agree on the positives; the 1-1 split on the safety criterion
        # is a tie, so it is not met and safety passes.
        assert result["score"] == 5 / 8
        assert result["safety_pass"] is True

    def test_unshuffling_maps_display_ordinals_back_to_rubric(self, scorer_mod):
        # Mark met only the criterion DISPLAYED first; after un-shuffling, the
        # met rubric index must be the first entry of the task's shuffle order.
        n = len(self.RUBRIC)
        order = scorer_mod._criterion_order("t5", n)
        grades = scorer_mod._parse_grades(_grades_json([True] + [False] * (n - 1)), n)
        met = [False] * n
        for position, grade in enumerate(grades):
            met[order[position]] = grade["met"]
        assert met[order[0]] is True
        assert sum(met) == 1
