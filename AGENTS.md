# Agent guide

Instructions for coding agents working in this repository. Human contributors:
see CONTRIBUTING.md.

## What this repository is

An evaluation harness for Arcophos health benchmarks in the HealthBench
Professional (HBP)-compatible format. Two independent implementations of one
scoring protocol:

- `src/arcophos_evals/` is the Inspect-native task pack (requires
  `inspect-ai`, Python 3.10+). Tasks resolve as `arcophos_evals/<task>`.
- `reference/hob_eval.py` is a zero-dependency reference runner (Python 3.9+
  standard library only) for environments where nothing can be installed.

The scoring formula lives in exactly one importable place,
`arcophos_evals.types.score_from_verdicts`. The reference runner mirrors it,
and `tests/test_parity.py` binds the two; if you change one side, the parity
test tells you what else must change.

## Commands

```bash
pip install -e ".[dev]"          # install with lint + test tooling
ruff check .                     # lint (pinned version in pyproject)
pytest tests/                    # package tests (needs inspect-ai)
python3 reference/test_hob_eval.py   # reference runner tests (stdlib only)
inspect eval arcophos_evals/hob_sample \
  --model <provider/model> --model-role grader=<provider/model> --limit 2
```

Use `-T source=tests/fixtures.jsonl` for a 2-task offline smoke, and
`mockllm/model` as either model for a no-network run.

## Rules that protect results

1. Never edit the versioned grading prompt (`GRADE_PROMPT`, defined in
   `src/arcophos_evals/_grading_prompt.py` and mirrored byte-identically in
   `reference/hob_eval.py`) without bumping its version constant in BOTH
   files; a parity test fails if the two copies drift. Published scores are
   only comparable under identical prompt versions.
2. Never change `score_from_verdicts` semantics. Any scoring change is a new
   task version (`TASK_VERSION` in `src/arcophos_evals/benchmarks/hob.py`)
   and a changelog entry.
3. The judge must stay blind: no point values and no reference answers in
   any grading prompt. Tests assert this; keep them passing.
4. Grading failures score NOANSWER with `grading_error` metadata. Never make
   a failure path produce a bare zero.
5. The pinned dataset digest (`SAMPLE_SHA256` in
   `src/arcophos_evals/dataset.py`) changes only together with a task-version
   bump when the published sample changes.

## Adding a benchmark

One module per benchmark in `src/arcophos_evals/benchmarks/`: its task
functions (datasets load from a local path; the canonical-download constants
in `dataset.py` are HOB-specific). Then import the tasks in
`benchmarks/__init__.py` and `_registry.py` (the `inspect_ai` entry point),
and add the names to `_TASKS` in the package root. The loader, scorer, and
reference runner are format-level and need no changes for a new benchmark in
the HBP-compatible shape.

## Checks before you finish

Run `ruff check .`, `pytest tests/`, and
`python3 reference/test_hob_eval.py`. All three must pass on a clean
checkout. If you touched the scorer or the reference runner, confirm
`tests/test_parity.py` still passes; it is the contract between the two
implementations.
