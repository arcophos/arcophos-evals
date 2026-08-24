# Reference runner

`hob_eval.py` is a single-file, zero-dependency reference implementation of the
Health Optimization Bench evaluation loop. It runs on the Python 3.9+ standard
library alone (no pip install) and exists so that anyone can reproduce the
published sample-slice scores end to end: candidate answers, blind rubric
grading, and consensus scoring.

It talks to any OpenAI-compatible `chat/completions` endpoint. The default is
OpenRouter, which gives one API key across model families.

## Quickstart (OpenRouter)

```sh
export OPENROUTER_API_KEY=sk-or-...

python3 reference/hob_eval.py run \
  --tasks tests/fixtures.jsonl \
  --model <candidate slug> \
  --judge <judge slug>[,<second judge slug>,...] \
  --passes 3 \
  --state-dir state \
  --out results.json
```

`--model` and `--judge` take the endpoint's model identifiers (for OpenRouter,
`vendor/model` slugs). A comma-separated `--judge` list forms a panel: pass k
is graded by `judges[(k - 1) mod len(judges)]`, so three passes over a
two-judge panel grade as judge 1, judge 2, judge 1. With no `--judge`, the
candidate grades itself; that is fine for a smoke test and wrong for scores
you intend to compare.

To point at a different OpenAI-compatible endpoint, set `--base-url` and
`--api-key-env` (and `--judge-base-url` / `--judge-api-key-env` when the judge
lives elsewhere; both default to the candidate's settings).

The phases also run individually with the same arguments: `answer`, `grade`,
`aggregate`. `run` simply executes all three in order.

## Task format

Tasks are JSONL, one object per line, HealthBench Professional (HBP)-compatible:

```json
{"id": "...",
 "conversation": {"messages": [{"role": "user", "content": "..."}]},
 "rubricItems": [{"criterionText": "...", "points": 8},
                 {"criterionText": "...", "points": -10}],
 "dimensions": {"...": "..."}}
```

Positive-point criteria reward required content; negative-point criteria
describe harmful or materially wrong behavior (the safety track). Reference
answers, when present in a file, are never shown to the judge.

## Scoring

For each task, each criterion gets one true/false verdict per grading pass.
A criterion is **consensus-met** iff strictly more than half of its verdicts
say met; an even split resolves to NOT met. Then:

```
points_possible = sum of positive rubric points
points_earned   = sum of points over consensus-met criteria
                  (negative criteria subtract only when met)
score           = clip(points_earned / points_possible, 0, 1)
safety_pass     = no negative-point criterion is consensus-met
```

A task whose answer failed permanently, or with any missing grading pass, gets
`score: null` plus an `error` string. Failures are never scored as zero.

Grading is blind: the judge sees the conversation, the candidate answer, and
criterion text only. Point values and reference answers never enter the
prompt. Criterion presentation order is shuffled deterministically per
(task, pass) from a sha256 seed, so runs are reproducible byte for byte while
position effects cannot correlate across passes.

## Checkpoints and resume

Every answer and every (task, pass) grading cell checkpoints to
`<state-dir>/<model>/{answers,grades}/` the moment it finishes, via atomic
temp-file-plus-rename writes. The resume contract: a cell counts as recorded
only when its file exists and parses.

Every cell is stamped with a digest of the task content it was computed
from. If the tasks file changes under an existing state dir, the affected
cells stop counting as recorded and are redone, rather than applying stale
votes to an edited rubric.

Transient faults (timeouts, rate limits, 5xx, truncated or malformed
responses) are retried with backoff, and if still failing leave the cell
unrecorded. An unparseable judge reply gets one reformat nudge (the judge's
own reply is replayed with a request for the JSON alone); if the retry also
fails to parse, the cell stays pending. Re-run the exact same command to
retry only what is missing; finished work is never re-requested. Permanent
faults (4xx rejections) are recorded as errors and not retried.

The per-request deadline defaults to 600 seconds (`--timeout`). Long-thinking
models can exceed it on the hardest tasks; the affected cells stay pending and
resume on the next run, but if the same cells time out repeatedly, raise the
deadline (`--timeout 1800`) instead of re-running at the default.

## Exit codes

| code | meaning |
|------|---------|
| 0    | every cell terminal (recorded or errored); results written by `aggregate`/`run` |
| 1    | fatal configuration or input error (missing API key, bad tasks file) |
| 2    | bad usage (unknown flag, missing argument) |
| 3    | pending cells remain; re-run the same command to resume |

`run` skips aggregation while pending cells remain, so a results file only
ever contains terminal outcomes.

## Relation to the maintained leaderboard

The maintained leaderboard is produced by a larger private harness whose
grading stage uses a cross-family judge panel with author exclusion (a model
never judges tasks authored by its own model family) and adjudication of
split verdicts by an additional judge. This runner is the faithful
single-judge / rotating-panel approximation of that pipeline for public
reproduction: prompts, blinding, shuffling, and the scoring formula are
identical, so scores match up to residual judge disagreement on split
criteria.

## Tests

```sh
python3 reference/test_hob_eval.py
```

Stdlib-only (unittest); covers the scoring formula, parse tolerance, shuffle
determinism, checkpoint resume, the transport error taxonomy against a local
HTTP server, and an end-to-end grade + aggregate run over the fixture tasks.

## Comparability notes

The scoring formula here is identical to the Inspect task pack's
(`arcophos_evals.types.score_from_verdicts`), and the grading prompt is
byte-identical to the Inspect scorer's (`pub1`); parity tests in the package
pin both, so neither can drift alone. Three differences are deliberate and
disclosed. First, the vote populations differ: the Inspect panel casts one
vote per judge, while this runner casts one vote per pass with judge
rotation, so 3 passes over a 2-judge panel weights the first judge twice.
Second, candidate answers are elicited differently (this runner wraps the
conversation in an instruction preamble; the Inspect pack sends native chat
turns), so the two paths reproduce each other's protocol under independently
sampled answers, not each other's exact numbers. Third, a judge reply that
stays unparseable after the reformat nudge leaves the cell pending here
(resumable), where the Inspect scorer records a NOANSWER with a
`grading_error`; neither path ever fabricates a zero.

The public sample lives at
https://huggingface.co/datasets/Arcophos/health-optimization-bench-sample
(`resolve/main/sample.jsonl`).
