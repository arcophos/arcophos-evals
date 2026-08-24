# Changelog

Task-affecting changes bump the relevant version constant
(`TASK_VERSION` in the benchmark module, `GRADE_PROMPT_VERSION` in the
grading prompt) and are recorded here. Corrections are published, never
applied silently.

## 0.1.0 (2026-08-24)

- Initial release: Inspect task pack for Health Optimization Bench
  (`hob_sample` plus ten micro-bench tasks, TASK_VERSION 1), blind rubric
  scorer with judge panels, and the zero-dependency reference runner. Both
  implementations grade with the byte-identical pub1 prompt
  (GRADE_PROMPT_VERSION), pinned to each other by parity tests.
- Public sample pinned to SHA-256
  `a3e8056fed1e9fa6dce059cc4144cadd9448859dc5e27139001753697d5d7d11`.
