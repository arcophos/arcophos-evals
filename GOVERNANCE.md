# Governance

A benchmark is only useful to people who have no reason to trust its operator. This document states who runs Health Optimization Bench, who pays for it, and the rules that hold regardless of either.

## Who runs the benchmark

Arcophos operates Health Optimization Bench: the task pipeline, the leaderboard, and this harness. Arcophos ships no models of its own. Its revenue comes from licensing task sets and running evaluations, which depends on the scores being accurate, not on any particular model winning.

## Funding

Arcophos is a venture-funded company. No AI model developer is an investor in Arcophos, funds this benchmark, or receives access to holdout content. If a model developer becomes a customer (for example, commissioning an evaluation of its own model), that relationship is disclosed alongside any score we publish for that developer's models. The policies below do not change with funding.

## Holdout policy

- 51 tasks are held out from every release.
- Access is restricted to the benchmark maintainers. Customers do not receive them. Funders do not receive them. Evaluated labs do not receive them. There is no price at which they are available.
- The holdout exists to detect overfitting to the released sets. Beginning with the next leaderboard release, each release publishes, per model, the gap between its public-set score and its holdout score.

## How leaderboard runs are configured

- The run configuration is committed before any scored run; starting with the next release, the configuration file and its hash are published in this repository before results: task versions, model ids, generation settings, and the grading panel.
- Graders are pinned. The panel's model ids and versions are frozen per release and recorded with the results.
- Every model in a release runs under the same configuration.
- The public-vs-holdout gap is published for every model beginning with the next release (see Holdout policy).

## Errata and versioning

- Every task carries a version. Any change that could affect a result, including prompt wording, rubric text, point values, gold answers, or citations, bumps the version.
- Corrections are published in an errata log: what changed, when, why, and which published results were affected.
- Nothing is corrected silently. If a published score changes, the change is announced and the prior value remains visible in the log.

## Conflicts of interest

- No pay-for-placement. Placement on the leaderboard cannot be bought.
- Labs never pay for scores: not for inclusion, not for re-runs, not for early access to results.
- A lab may request evaluation of its model. If we run it, the result is published with a note that the run was requested, under the same pre-committed configuration as every other run.
- Any exception to these rules would itself be disclosed on the leaderboard.

## Contact

Questions, disputes, or reports of a violation of this document: [info@arcophos.com](mailto:info@arcophos.com).
