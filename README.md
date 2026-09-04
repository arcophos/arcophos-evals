<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/arcophos-logo-dark.svg">
    <img src="assets/arcophos-logo.svg" width="240" alt="Arcophos">
  </picture>
</p>

<h1 align="center">Arcophos Evals</h1>

<p align="center">Evaluation harness for Arcophos health benchmarks in the HealthBench Professional compatible format</p>

<p align="center">
  <img alt="Code license: MIT" src="https://img.shields.io/badge/code-MIT-blue">
  <img alt="Sample data: CC BY-NC 4.0" src="https://img.shields.io/badge/sample%20data-CC%20BY--NC%204.0-lightgrey">
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="Inspect task pack" src="https://img.shields.io/badge/Inspect-task%20pack-6a4fb3">
</p>

Health Optimization Bench (HOB) tests whether language models get current clinical evidence right: which trial studied which drug, at what dose, with what result, and what the guidelines actually say. Every task is graded criterion by criterion against a written rubric, with a separate safety verdict for the one error the task treats as disqualifying. This repository is the open harness: an [Inspect](https://inspect.aisi.org.uk/)-native task pack, the grading protocol used on the public leaderboard, and a reference runner with no dependencies.

Health Optimization Bench is the first registered benchmark. Every future Arcophos benchmark
ships in this same harness as a new task module: the loader, scorer, and reference runner are
format-level, so a benchmark release adds tasks, never infrastructure.

## Quickstart

Requires Python 3.10+ and provider API keys in the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and so on).

```bash
pip install "git+https://github.com/arcophos/arcophos-evals.git"
```

Run the public 30-task sample:

```bash
inspect eval arcophos_evals/hob_sample --model openai/gpt-4o
```

Run a single micro bench:

```bash
inspect eval arcophos_evals/hob_lipids_ascvd --model anthropic/claude-opus-5
```

Each micro-bench task filters the public sample to its micro bench (3 of the
30 sample tasks by default; pass `-T source=` to run a licensed or local file
through the same filter).

| Inspect task | Filters to micro bench |
|---|---|
| `hob_sample` | all 10 (the full 30-task sample) |
| `hob_blood_pressure` | Blood Pressure Optimization |
| `hob_cancer_screening` | Cancer Screening & Early Detection |
| `hob_exercise_fitness` | Exercise & Cardiorespiratory Fitness |
| `hob_geroscience` | Longevity / Geroscience Pharmacology |
| `hob_hormone_optimization` | Hormone Optimization |
| `hob_incretin_clinical` | Incretin Therapeutics: Clinical Decisions |
| `hob_incretin_evidence` | Incretin Therapeutics: Evidence Synthesis |
| `hob_lipids_ascvd` | Lipids & ASCVD Prevention |
| `hob_nutrition_supplements` | Nutrition & Supplements |
| `hob_sleep` | Sleep Optimization |

Grading uses a pinned default judge, recorded in `src/arcophos_evals/scorer.py`. Override it for a run:

```bash
inspect eval arcophos_evals/hob_sample --model openai/gpt-4o \
  --model-role grader=anthropic/claude-opus-5
```

Or grade with a panel. Each criterion is then decided by majority vote:

```bash
inspect eval arcophos_evals/hob_sample --model openai/gpt-4o \
  -T judge="anthropic/claude-opus-5,openai/gpt-5.6-sol,google/gemini-3-pro"
```

### Options

| Flag | Effect |
|---|---|
| `--limit N` | Run the first N tasks only |
| `-T source=path.jsonl` | Evaluate a local HBP-compatible file (licensed full sets) instead of the public sample |
| `-T judge="model"` | Single judge model for grading |
| `-T judge="a,b,c"` | Judge panel; per-criterion majority vote, ties are not met |
| `--model-role grader=model` | Judge via Inspect model roles; an explicit `-T judge=` takes precedence |
| `-T cache=false` | Resample candidate answers instead of reusing inspect_ai's generation cache (the default `cache=true` makes repeat runs reuse identical completions, which is wrong for run-to-run variance measurement) |
| `--epochs 1` | Keep the default; the safety and graded-mean metrics read per-sample metadata, which Inspect does not reduce across epochs |

## Zero-dependency reproduction

[`reference/hob_eval.py`](reference/hob_eval.py) reruns the same tasks under the same blinding rules and scoring formula using only the Python 3.9 standard library. There is nothing to install. It grades with the byte-identical prompt the Inspect scorer uses (`pub1`; a parity test pins the two template strings together), and both record the prompt version in results. It works against any OpenAI-compatible API, for example OpenRouter:

```bash
export OPENROUTER_API_KEY=sk-or-...
curl -LO https://huggingface.co/datasets/Arcophos/health-optimization-bench-sample/resolve/main/sample.jsonl
python3 reference/hob_eval.py run --tasks sample.jsonl --model openai/gpt-4o --judge anthropic/claude-opus-5
```

CI runs `reference/test_hob_eval.py` on a bare Python 3.9 with no packages installed, so the runner stays dependency-free by construction.

## The benchmark

16 frontier models from 12 labs are evaluated. The leader scores 70.9 of 100 and the field spans 66 points. Top five as of September 2026 (the live leaderboard supersedes this snapshot):

| Rank | Model | Lab | Score |
|---:|---|---|---:|
| 1 | Claude Fable 5 | Anthropic | 70.9 |
| 2 | Claude Opus 5 | Anthropic | 69.3 |
| 3 | Grok 4.6 | xAI | 66.8 |
| 4 | GPT-5.6 Sol (max) | OpenAI | 66.6 |
| 5 | GPT-5.6 Sol (high) | OpenAI | 64.6 |

Claude Fable 5.1, added on 2026-09-04, scores 47.3 as served on this set because its safeguard classifier declined 97 of the 257 tasks; over the tasks it answered its mean is 75.9, and it leads the incretin evidence suite at 84.9. A declined task is graded like any other completion and earns no credit. The full leaderboard, per-micro-bench results, confidence intervals, and declined-task counts are at [healthoptimizationbench.com](https://healthoptimizationbench.com).

977 tasks have been authored across 10 micro benches. 346 are released: the 89-task Incretin Evidence ranking set and 257 tasks across eight subject suites (every micro bench above except the two incretin sets; Incretin Clinical Decisions appears in the public sample but is not yet a released ranking set). A further 51 tasks form a confidential holdout that is never released (see [GOVERNANCE.md](GOVERNANCE.md)).

## How tasks are built

| Stage | Rule |
|---|---|
| Authoring | Five frontier model families draft tasks independently from primary sources. |
| Gold and citation audit | Model families other than the author verify each gold answer and citation. |
| Removal | A task is dropped only when two families independently confirm the same defect. |
| Difficulty banding | Two-sided: tasks the field always gets right, or that no model can resolve, are cut. |
| Clinical validation | Clinical tasks are validated by clinicians. |
| Grading | Blind three-family panel with the author family excluded. Split verdicts escalate to a fourth family. |
| Holdout | 51 tasks are withheld from release and used to check for overfitting. |

## What this harness does differently

Compared with the reference implementations this format descends from (openai/simple-evals and the community HealthBench port in inspect_evals):

| Practice | Here | Elsewhere |
|---|---|---|
| Judge sees point values | Never | Both show `[points]` in the grader prompt |
| Failed grading | Scores NOANSWER with `grading_error` metadata; `mean_of_graded` exposes the delta | Retried forever, or silently scored as a model failure |
| Rubric position effects | All criteria in one judge call, deterministically shuffled per task | One judge call per criterion (no shared-prompt position effects, at N times the judge cost) |
| Multi-judge panels | Per-criterion majority vote, with degraded panels recorded (`judges_requested` vs `judges`) | Single judge only |
| Cross-implementation drift | Parity tests bind the Inspect scorer and the reference runner to identical scoring (formula, grading prompt, and consensus rule) | No second implementation |

## Scoring

Each task carries a rubric of criteria worth between -10 and 10 points, never 0. Exactly one criterion per task is negative: the safety criterion. It names the specific failure that makes an answer unsafe or disqualifying for that task, such as asserting a fabricated trial result as fact.

A judge reads the model's answer and marks each criterion met or not met. The judge never sees the reference answer.

```
score = clip(points_earned / positive_points_possible, 0, 1)
```

Positive criteria add points when met. The safety criterion subtracts its points only when the model commits the flagged error. `safety_pass` is reported alongside the score as a separate boolean, true when the safety criterion was not committed, and is never folded into the score. Leaderboard scores are this ratio times 100, averaged over tasks.

Runs from this repository reproduce the public sample slice. The maintained leaderboard runs the full released sets under a frozen cross-family grading panel, so a single-judge sample run will differ somewhat from published numbers.


### Worked example

A task with criteria worth +10, +9, +8, and one safety criterion at -8. The judge marks the
first two met, the third not met, and the safety criterion committed:

```
points_possible = 10 + 9 + 8      = 27   (positive criteria only)
points_earned   = 10 + 9 - 8      = 11   (safety subtracts when committed)
score           = clip(11/27, 0, 1) = 0.407
safety_pass     = false               (reported separately, never folded in)
```

## Data

The public sample is 30 tasks, 3 per micro bench, published on Hugging Face as [`Arcophos/health-optimization-bench-sample`](https://huggingface.co/datasets/Arcophos/health-optimization-bench-sample) under CC BY-NC 4.0.

Each record:

| Field | Type | Notes |
|---|---|---|
| `id` | string | Stable task id. |
| `conversation.messages` | list of `{role, content}` | The prompt. Always ends on a user turn. |
| `dimensions.micro_bench` | string | Micro bench name. |
| `dimensions.use_case` | string | Intended setting, for example `research`. |
| `dimensions.type` | string | Task framing, for example `good_faith`. |
| `dimensions.difficulty` | string | Difficulty band. |
| `rubricItems` | list of `{criterionText, points}` | Points in -10..10, never 0. Exactly one negative item per task. |
| `physicianResponse` | string | Reference answer. Never shown to graders. |

The full released sets (346 tasks) are licensed for model evaluation. Write to [info@arcophos.com](mailto:info@arcophos.com).

## What is public and what is not

| Item | Status |
|---|---|
| Harness code (this repository) | Public. MIT. |
| Grading prompts and scoring code | Public. In this repository. |
| Methodology | Public. This README and [healthoptimizationbench.com](https://healthoptimizationbench.com). |
| 30-task sample | Public. CC BY-NC 4.0. |
| Full released sets (346 tasks) | Licensed. Contact [info@arcophos.com](mailto:info@arcophos.com). |
| Holdout (51 tasks) | Never released, to anyone. |

## Evaluation report

Live replication runs of this harness over the 30-task public sample, August and September
2026. All runs used a single-judge panel (`gpt-4.1` via OpenRouter; the packaged default
resolves the same model through the OpenAI API), one grading pass, judge temperature 0.

| model | accuracy | stderr | safety_pass | n | run date |
|---|---|---|---|---|---|
| `anthropic/claude-fable-5.1` | 0.506 | 0.080 | 0.967 | 30 | 2026-09-01 |
| `moonshotai/kimi-k3` | 0.570 | 0.076 | 0.933 | 30 | 2026-08-24 |
| `openai/gpt-4o-mini` | 0.108 | 0.029 | 0.333 | 30 | 2026-08-24 |

The Claude Fable 5.1 row was run on its release day through OpenRouter with the model's
default sampling and no system prompt, which is the harness's plain chat elicitation. On 12
of the 30 tasks the API returned a content-filter stop with a refusal notice instead of an
answer (Anthropic's Fable 5.1 safeguard classifier, category `bio`). Those samples are
scored by the rubric like any other completion and earn zero credit, so the 0.506 figure
is the score of the model as served; over the 18 answered tasks the mean is 0.844. The
refusals are deterministic per prompt and reproduce through the Claude Code CLI. A
sample-slice number under one judge does not stand in for the leaderboard entry, which is
elicited and graded under the full protocol.

For context, the published leaderboard at
[healthoptimizationbench.com](https://healthoptimizationbench.com) scores kimi-k3 at 59.9 on
the full released task set under the stricter three-judge, three-pass protocol. A 30-task
sample with one judge is not expected to reproduce that number exactly; it lands within its
sampling error.

Cross-implementation agreement, measured on live traffic rather than asserted:

- **Same answers, both scorers.** The reference runner regraded the identical 30 candidate
  answers from the Inspect run above: mean 0.580 vs 0.570, with 23 of 30 per-task scores
  byte-identical (median |difference| 0.000, mean 0.051, max 0.340) and safety verdicts
  agreeing on 28 of 30 tasks. Residual differences come from judge nondeterminism and the
  deliberately independent criterion shuffle, not from the scoring code; the parity tests
  pin the formula, the grading prompt, and the consensus rule across both implementations.
- **Independent end to end.** The reference runner also ran the whole pipeline itself,
  eliciting its own kimi-k3 answers. Over the 27 tasks that completed under the default
  request deadline it scored 0.709 against the Inspect estimate of 0.578 on the same tasks,
  a gap of about one combined standard error, consistent with independently sampled answers
  under the disclosed elicitation difference (instruction preamble vs native chat turns).

## Citation

```bibtex
@misc{healthoptimizationbench2026,
  title        = {Health Optimization Bench},
  author       = {{Arcophos}},
  year         = {2026},
  howpublished = {\url{https://healthoptimizationbench.com}}
}
```

## License

Code in this repository is MIT licensed. The sample dataset is CC BY-NC 4.0. The full released task sets and the holdout are covered by neither license.
