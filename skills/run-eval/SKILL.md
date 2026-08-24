---
name: run-eval
description: Run a Health Optimization Bench evaluation against any model with the Inspect task pack or the zero-dependency reference runner, and read the results.
---

# Run an evaluation

## Inspect path (recommended)

1. Install: `pip install "git+https://github.com/arcophos/arcophos-evals.git"`
   (or `pip install -e .` from a checkout).
2. Pick a candidate model and a judge. Any Inspect model string works
   (`openai/...`, `anthropic/...`, `openrouter/...`, `mockllm/model` for a
   dry run). Set the provider's API key environment variable.
3. Run the full sample or one micro bench:

```bash
inspect eval arcophos_evals/hob_sample \
  --model openrouter/moonshotai/kimi-k3 \
  --model-role grader=openrouter/openai/gpt-4.1
inspect eval arcophos_evals/hob_sleep --model <model> --model-role grader=<judge>
```

Task names: `hob_sample` plus one task per micro bench; list them with
`python -c "import arcophos_evals; print([t for t in dir(arcophos_evals) if t.startswith('hob_')])"`.

4. Options: `--limit N` for a subset, `-T source=path.jsonl` for a licensed
   full set, `-T judge="a,b,c"` for a majority-vote panel, `--epochs 1`
   always (metrics read per-sample metadata).
5. Read results from the `.eval` log: `inspect view` for a UI, or
   programmatically via `inspect_ai.log.read_eval_log(path)`. The metrics are
   `accuracy` (mean score), `stderr`, `safety_pass_metric` (fraction with no
   committed safety criterion), and `mean_of_graded` (mean over successfully
   graded samples). Per-sample metadata carries per-criterion verdicts.

## Zero-dependency path

No installs, Python 3.9+:

```bash
curl -LO https://huggingface.co/datasets/Arcophos/health-optimization-bench-sample/resolve/main/sample.jsonl
python3 reference/hob_eval.py run --tasks sample.jsonl \
  --model <model-id> --base-url <openai-compatible-url> --api-key-env <ENV_VAR> \
  --judge <judge-id> --state-dir state --out results.json
```

Exit code 3 means pending work remains (rate limits, timeouts, 5xx): rerun the same command;
checkpoints make resumption free. Results JSON contains per-task scores,
`safety_pass`, and a summary block.

## Interpreting scores

`score = clip(points_earned / positive_points, 0, 1)` per task. The single
negative-point criterion subtracts only when the model commits the described
misstatement; `safety_pass` reports that separately and is never folded into
the score. Failed grading is score `null`/NOANSWER, never zero.
