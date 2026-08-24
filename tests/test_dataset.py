"""Unit tests for ``arcophos_evals.dataset``.

``inspect_ai`` is deliberately not imported at module scope: the loader must
work standalone. Only the ``to_inspect_dataset`` tests touch Inspect, guarded
by ``pytest.importorskip``.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from arcophos_evals.dataset import filter_micro_bench, load_tasks, to_inspect_dataset
from arcophos_evals.types import RubricItem

FIXTURES = Path(__file__).parent / "fixtures.jsonl"


def make_record(**overrides: object) -> dict:
    """Return a minimal well-formed HBP record; ``overrides`` replace top-level fields."""
    record: dict = {
        "id": "task-0",
        "conversation": {"messages": [{"role": "user", "content": "Question?"}]},
        "dimensions": {"micro_bench": "Sleep Optimization", "difficulty": "easy"},
        "rubricItems": [
            {"criterionText": "States the answer.", "points": 5},
            {"criterionText": "Recommends something unsafe.", "points": -4},
        ],
        "physicianResponse": "Answer.",
    }
    record.update(overrides)
    return record


def write_jsonl(path: Path, records: list[dict]) -> str:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    return str(path)


def test_load_tasks_from_fixtures() -> None:
    tasks = load_tasks(str(FIXTURES))
    assert [t.task_id for t in tasks] == [
        "32939c021f4d1a21ede48f39822f4f53",
        "3aa3371f62dc43632b92e0f4f18022db",
    ]


def test_field_mapping() -> None:
    first = load_tasks(str(FIXTURES))[0]
    assert first.messages[0]["role"] == "user"
    assert first.messages[0]["content"].startswith("Please audit the following excerpt")
    assert first.micro_bench == "Blood Pressure Optimization"
    # micro_bench is promoted to its own field; the remaining dimensions are kept.
    assert first.dimensions == {
        "use_case": "research",
        "type": "good_faith",
        "difficulty": "difficult",
    }
    assert first.reference_answer is not None
    assert first.reference_answer.startswith("The excerpt contains multiple factual errors")


def test_rubric_parsing() -> None:
    first, second = load_tasks(str(FIXTURES))
    assert len(first.rubric) == 6
    assert len(second.rubric) == 7
    assert isinstance(first.rubric[0], RubricItem)
    assert first.rubric[0].criterion_text.startswith("Correctly identifies that the agent")
    assert first.rubric[0].points == 8
    safety = [item for item in first.rubric if item.is_safety]
    assert len(safety) == 1
    assert safety[0].points == -10


def test_fixtures_satisfy_invariants() -> None:
    # Both fixture records are well formed: strict mode passes and nothing warns.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tasks = load_tasks(str(FIXTURES), strict=True)
    assert len(tasks) == 2


def test_micro_bench_defaults_to_empty_string(tmp_path: Path) -> None:
    record = make_record(dimensions={"difficulty": "easy"})
    tasks = load_tasks(write_jsonl(tmp_path / "tasks.jsonl", [record]))
    assert tasks[0].micro_bench == ""
    assert tasks[0].dimensions == {"difficulty": "easy"}


def test_zero_negative_items_warns_but_loads(tmp_path: Path) -> None:
    record = make_record(rubricItems=[{"criterionText": "States the answer.", "points": 5}])
    path = write_jsonl(tmp_path / "tasks.jsonl", [record])
    with pytest.warns(UserWarning, match="0 negative-point"):
        tasks = load_tasks(path)
    assert len(tasks) == 1  # third-party HBP files stay loadable


def test_multiple_negative_items_warn_but_load(tmp_path: Path) -> None:
    record = make_record(
        rubricItems=[
            {"criterionText": "States the answer.", "points": 5},
            {"criterionText": "Recommends something unsafe.", "points": -4},
            {"criterionText": "Omits a red flag.", "points": -2},
        ]
    )
    path = write_jsonl(tmp_path / "tasks.jsonl", [record])
    with pytest.warns(UserWarning, match="2 negative-point"):
        tasks = load_tasks(path)
    assert len(tasks) == 1


def test_strict_mode_raises_on_invariant_violation(tmp_path: Path) -> None:
    record = make_record(rubricItems=[{"criterionText": "States the answer.", "points": 5}])
    path = write_jsonl(tmp_path / "tasks.jsonl", [record])
    with pytest.raises(ValueError, match="0 negative-point"):
        load_tasks(path, strict=True)


def test_conversation_must_end_on_user_turn(tmp_path: Path) -> None:
    record = make_record(
        conversation={
            "messages": [
                {"role": "user", "content": "Question?"},
                {"role": "assistant", "content": "Answer."},
            ]
        }
    )
    path = write_jsonl(tmp_path / "tasks.jsonl", [record])
    with pytest.warns(UserWarning, match="ends on a 'assistant' turn"):
        load_tasks(path)
    with pytest.raises(ValueError, match="ends on a 'assistant' turn"):
        load_tasks(path, strict=True)


def test_empty_rubric_warns(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "tasks.jsonl", [make_record(rubricItems=[])])
    with pytest.warns(UserWarning, match="rubric has no items"):
        load_tasks(path)


def test_missing_required_field_raises(tmp_path: Path) -> None:
    record = make_record()
    del record["id"]
    path = write_jsonl(tmp_path / "tasks.jsonl", [record])
    with pytest.raises(ValueError, match="not an HBP-compatible record"):
        load_tasks(path)


def test_invalid_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_tasks(str(path))


def test_missing_file_raises() -> None:
    with pytest.raises(ValueError, match="dataset file not found"):
        load_tasks("/nonexistent/hob-tasks.jsonl")


def test_filter_micro_bench() -> None:
    tasks = load_tasks(str(FIXTURES))
    assert filter_micro_bench(tasks, "Blood Pressure Optimization") == tasks
    assert filter_micro_bench(tasks, "Sleep Optimization") == []


def test_to_inspect_dataset() -> None:
    pytest.importorskip("inspect_ai")
    from inspect_ai.model import ChatMessageUser

    tasks = load_tasks(str(FIXTURES))
    dataset = to_inspect_dataset(tasks)
    assert len(dataset) == 2
    sample = dataset[0]
    assert sample.id == tasks[0].task_id
    assert [type(m) for m in sample.input] == [ChatMessageUser]
    assert sample.input[0].content == tasks[0].messages[0]["content"]
    assert sample.metadata is not None
    assert sample.metadata["micro_bench"] == "Blood Pressure Optimization"
    assert sample.metadata["dimensions"] == tasks[0].dimensions
    assert sample.metadata["rubric"][0] == {
        "criterion_text": tasks[0].rubric[0].criterion_text,
        "points": 8,
    }
    assert {item["points"] < 0 for item in sample.metadata["rubric"]} == {True, False}


def test_to_inspect_dataset_maps_system_and_rejects_unknown_role(tmp_path: Path) -> None:
    pytest.importorskip("inspect_ai")
    from inspect_ai.model import ChatMessageSystem

    record = make_record(
        conversation={
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "Question?"},
            ]
        }
    )
    tasks = load_tasks(write_jsonl(tmp_path / "tasks.jsonl", [record]))
    dataset = to_inspect_dataset(tasks)
    assert isinstance(dataset[0].input[0], ChatMessageSystem)

    bad = make_record(
        conversation={"messages": [{"role": "tool", "content": "output"}]}
    )
    with pytest.warns(UserWarning, match="ends on a 'tool' turn"):
        tasks = load_tasks(write_jsonl(tmp_path / "bad.jsonl", [bad]))
    with pytest.raises(ValueError, match="unsupported message role 'tool'"):
        to_inspect_dataset(tasks)
