# Security

Report vulnerabilities privately to info@arcophos.com or through GitHub's
private vulnerability reporting on this repository. We acknowledge reports
within 72 hours.

Scope notes for evaluators running this harness:

- The harness sends task content and model answers only to the model
  endpoints you configure. Nothing is transmitted to Arcophos.
- API keys are read from environment variables only. They are never written
  to logs, checkpoints, or result files.
- The reference runner writes checkpoints and results to paths you specify;
  treat those directories as containing model outputs you may not want to
  publish.

Disputes about task content or grading are not security issues; open a
regular issue as described in CONTRIBUTING.md.
