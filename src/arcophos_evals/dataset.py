"""Load HBP-compatible JSONL task files and convert them for Inspect.

- The canonical public sample is downloaded with ``urllib`` only (no third-party
  HTTP or datasets dependency) and cached under the user cache directory keyed by
  the pinned content digest.
- Validation warns rather than raises by default: third-party HBP-format files
  legitimately contain zero or multiple negative-point rubric items. Pass
  ``strict=True`` to enforce the HOB invariants instead.
- ``inspect_ai`` is imported lazily inside :func:`to_inspect_dataset` so the
  loader works without Inspect installed.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

from arcophos_evals.types import HOBTask, RubricItem

if TYPE_CHECKING:  # pragma: no cover - typing only
    from inspect_ai.dataset import MemoryDataset

SAMPLE_URL = (
    "https://huggingface.co/datasets/Arcophos/health-optimization-bench-sample"
    "/resolve/main/sample.jsonl"
)
_CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "arcophos-evals"
_DOWNLOAD_TIMEOUT_SECONDS = 30.0

SAMPLE_SHA256 = "a3e8056fed1e9fa6dce059cc4144cadd9448859dc5e27139001753697d5d7d11"
"""Pinned digest of the canonical sample. Updated, with a task-version bump,
whenever the published sample changes; a mismatch means the download does not
match what the published scores were computed on."""


def load_tasks(source: str | None = None, *, strict: bool = False) -> list[HOBTask]:
    """Parse an HBP-compatible JSONL file into :class:`HOBTask` objects.

    Args:
        source: Path to a local JSONL file. When ``None``, the canonical public
            sample is downloaded from :data:`SAMPLE_URL` (cached after first use);
            a download that does not match :data:`SAMPLE_SHA256` raises ``OSError``.
        strict: When ``True``, records that violate the HOB invariants (the
            conversation ends on a user turn, at least one rubric item, exactly
            one negative-point item) raise :class:`ValueError`. When ``False``
            (the default) such records produce a :class:`UserWarning` and are
            kept, so third-party HBP-format files still load.

    Returns:
        One :class:`HOBTask` per JSONL record, in file order.

    Raises:
        ValueError: On unreadable files, invalid JSON, or records missing
            required fields, and, with ``strict=True``, on invariant
            violations.
        OSError: When the canonical sample cannot be downloaded.
    """
    path = Path(source).expanduser() if source is not None else _cached_sample_path()
    if not path.is_file():
        raise ValueError(
            f"dataset file not found: {path}; pass load_tasks(source=...) with a "
            "path to an HBP-compatible JSONL file"
        )
    tasks: list[HOBTask] = []
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            origin = f"{path}:{lineno}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{origin}: invalid JSON: {exc}") from exc
            hob_task = _parse_record(record, origin)
            _validate(hob_task, origin, strict=strict)
            tasks.append(hob_task)
    if not tasks:
        raise ValueError(f"{path} contains no task records")
    return tasks


def filter_micro_bench(tasks: list[HOBTask], name: str) -> list[HOBTask]:
    return [hob_task for hob_task in tasks if hob_task.micro_bench == name]


def to_inspect_dataset(tasks: list[HOBTask]) -> MemoryDataset:
    """Convert tasks to an ``inspect_ai`` :class:`MemoryDataset`.

    Each sample's input is the conversation as Inspect chat messages, its id is
    the task id, and its metadata carries the rubric (as a list of
    ``{criterion_text, points}`` dicts), the micro bench name, and the remaining
    dimensions. Reference answers are omitted: never shown to graders.
    """
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser

    message_types = {
        "user": ChatMessageUser,
        "assistant": ChatMessageAssistant,
        "system": ChatMessageSystem,
    }
    samples = []
    for hob_task in tasks:
        input_messages = []
        for message in hob_task.messages:
            role = message.get("role")
            if role not in message_types:
                raise ValueError(
                    f"task {hob_task.task_id!r}: unsupported message role {role!r}; "
                    "HBP conversations contain only 'system', 'user', and 'assistant' turns"
                )
            input_messages.append(message_types[role](content=message["content"]))
        samples.append(
            Sample(
                input=input_messages,
                id=hob_task.task_id,
                metadata={
                    "rubric": [
                        {"criterion_text": item.criterion_text, "points": item.points}
                        for item in hob_task.rubric
                    ],
                    "micro_bench": hob_task.micro_bench,
                    "dimensions": hob_task.dimensions,
                },
            )
        )
    return MemoryDataset(samples)


def _cached_sample_path() -> Path:
    """Return a local path to the canonical sample, downloading it on first use."""
    # Keyed by the pinned digest: bumping SAMPLE_SHA256 orphans stale caches
    # instead of silently serving an outdated sample.
    cache_path = _CACHE_DIR / f"{SAMPLE_SHA256}.jsonl"
    if cache_path.is_file():
        # The filename carries the digest, but the bytes are re-verified: a
        # truncated or tampered cache file must trigger a re-download, not a run
        # over silently different data.
        if hashlib.sha256(cache_path.read_bytes()).hexdigest() == SAMPLE_SHA256:
            return cache_path
        cache_path.unlink()
    try:
        with urllib.request.urlopen(SAMPLE_URL, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
            payload = response.read()
    except OSError as exc:  # URLError/HTTPError are OSError subclasses
        raise OSError(
            f"could not download the canonical sample from {SAMPLE_URL} ({exc}); "
            "check your network connection, or pass load_tasks(source=...) with "
            "a local JSONL path"
        ) from exc
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SAMPLE_SHA256:
        raise OSError(
            f"downloaded sample does not match the pinned digest "
            f"(got {digest[:12]}..., expected {SAMPLE_SHA256[:12]}...); the published "
            "sample may have been updated: upgrade arcophos-evals, or pass "
            "load_tasks(source=...) to evaluate a local file"
        )
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=_CACHE_DIR, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as tmp_file:
            tmp_file.write(payload)
        Path(tmp_name).replace(cache_path)  # atomic: never leaves a truncated cache file
    finally:
        Path(tmp_name).unlink(missing_ok=True)
    return cache_path


def _parse_record(record: object, origin: str) -> HOBTask:
    if not isinstance(record, dict):
        # Data-validation failure on file content, not an API-misuse TypeError.
        raise ValueError(  # noqa: TRY004
            f"{origin}: expected a JSON object, got {type(record).__name__}"
        )
    try:
        task_id = str(record["id"])
        messages = [dict(message) for message in record["conversation"]["messages"]]
        rubric = [
            RubricItem(criterion_text=item["criterionText"], points=item["points"])
            for item in record["rubricItems"]
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{origin}: not an HBP-compatible record ({exc!r}); expected fields are "
            "id, conversation.messages with role/content, and rubricItems with "
            "criterionText/points"
        ) from exc
    dimensions = dict(record.get("dimensions") or {})
    return HOBTask(
        task_id=task_id,
        messages=messages,
        rubric=rubric,
        micro_bench=dimensions.pop("micro_bench", ""),
        dimensions=dimensions,
        reference_answer=record.get("physicianResponse"),
    )


def _validate(hob_task: HOBTask, origin: str, *, strict: bool) -> None:
    """Check the HOB invariants; raise when ``strict``, warn otherwise."""
    problems: list[str] = []
    if not hob_task.messages:
        problems.append("conversation has no messages")
    elif hob_task.messages[-1].get("role") != "user":
        last_role = hob_task.messages[-1].get("role")
        problems.append(f"conversation ends on a {last_role!r} turn instead of 'user'")
    if not hob_task.rubric:
        problems.append("rubric has no items")
    else:
        negatives = sum(1 for item in hob_task.rubric if item.points < 0)
        if negatives != 1:
            problems.append(
                f"rubric has {negatives} negative-point (safety) items instead of exactly 1"
            )
        if not any(item.points > 0 for item in hob_task.rubric):
            # score_from_verdicts divides by the positive total; a rubric with no
            # positive criteria cannot be scored at all.
            problems.append("rubric has no positive-point criteria")
        if any(item.points == 0 or not -10 <= item.points <= 10 for item in hob_task.rubric):
            problems.append("rubric points must be integers in -10..10 and never 0")
    if not problems:
        return
    message = f"{origin} (task {hob_task.task_id!r}): " + "; ".join(problems)
    if strict:
        raise ValueError(message)
    warnings.warn(message, stacklevel=3)
