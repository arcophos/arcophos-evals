# Contributing

## Issues

Bug reports and questions are welcome.

To dispute a task's gold answer or rubric, open an issue with the task id and a citation to the primary source. Disputed tasks get the same treatment as tasks in construction: two model families other than the author re-audit the task, and it is corrected or removed only when both confirm the defect. The outcome is posted in the issue and, if a published result was affected, in the errata log (see [GOVERNANCE.md](GOVERNANCE.md)).

## Pull requests

In scope: harness code, provider adapters, tests, and documentation.

Out of scope: task content. Changes to tasks are made only by the maintainers, through the dispute process above, so that every content change carries a version bump and an audit trail.

## Development setup

```bash
pip install -e ".[dev]"
ruff check .
pytest tests/
```

The reference runner must stay standard-library-only on Python 3.9. Its self-test runs without installing anything:

```bash
python3 reference/test_hob_eval.py
```

## AI assistance

For large pull requests, please state whether and how AI tools were used to produce the change. Disclosure does not affect acceptance. It helps us review with the right eyes.
