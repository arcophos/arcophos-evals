---
name: add-benchmark
description: Add a new HBP-format benchmark to this harness as a task module, without touching the loader, scorer, or reference runner.
---

# Add a benchmark

Prerequisite: the benchmark's tasks are in the HealthBench Professional
compatible JSONL shape (one object per line: `id`, `conversation.messages`
ending on a user turn, `rubricItems` of `{criterionText, points}` with integer
points in -10..10, never 0, exactly one negative item, optional
`physicianResponse`, `dimensions` including `micro_bench`).

1. Produce the dataset as a local JSONL file. The loader takes local paths;
   the canonical-download path (URL plus pinned SHA-256) is HOB-specific, so
   a new benchmark either ships its own download constant pair in its module
   or documents a download command for users.
2. Create `src/arcophos_evals/benchmarks/<name>.py` modeled on `hob.py`:
   a display-name mapping, a `TASK_VERSION`, and `@task` functions calling
   the shared factory pattern with `load_tasks(source)` and
   `rubric_scorer(judge=judge)`. Every task takes `source` and `judge`.
3. Register: import the tasks in `benchmarks/__init__.py` and in
   `_registry.py`; add the names to `_TASKS` in the package root.
4. Tests: add a fixtures file (2 tasks) with a license note if the data is
   not MIT, plus a loader test and one mockllm integration test following
   `tests/test_scorer_integration.py`.
5. Docs: add the tasks to the README table and a CHANGELOG.md entry.
6. Verify: `ruff check .`, `pytest tests/`, and
   `inspect eval arcophos_evals/<new_task> --model mockllm/model
   --model-role grader=mockllm/model -T source=<fixtures> --limit 1`.
