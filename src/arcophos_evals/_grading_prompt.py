"""Grading prompt for the public HOB rubric scorer.

The prompt is versioned so every Score records exactly which grading
instructions produced it: bump :data:`GRADE_PROMPT_VERSION` whenever
:data:`GRADE_PROMPT` changes, because even small wording changes can shift
judge behaviour and break comparability between runs.

The prompt is deliberately blind. The judge sees the conversation, the
candidate answer, and criterion text only; point values and the reference
answer are never rendered. See :mod:`arcophos_evals.scorer` for the protocol
rationale.
"""
from __future__ import annotations

GRADE_PROMPT_VERSION = "pub1"
"""Version tag for the grading prompt below; recorded in every Score's metadata."""

GRADE_PROMPT = """\
You are grading one candidate answer for a health-optimization benchmark.

Below are the conversation the candidate was responding to, the candidate's answer, and a
numbered list of grading criteria. For each criterion, decide whether the candidate answer
meets it. Judge only the candidate answer against the criterion text; do not reward effort,
verbosity, or content the criteria do not ask for.

<conversation>
{conversation}
</conversation>

<candidate_answer>
{candidate_answer}
</candidate_answer>

<criteria>
{criteria}
</criteria>

Grading rules:
- "met" is true only when the answer clearly satisfies the criterion as written.
- A criterion that describes an error, omission, or harmful content is met when the answer
  commits what the criterion describes.
- No partial credit: when in doubt, the criterion is not met.

Respond with STRICT JSON only -- no markdown fences, no surrounding text -- in exactly this
shape:
{{"grades": [{{"ordinal": 1, "met": true, "explanation": "<max 40 words>"}}, ...]}}
Include every ordinal from 1 to {n} exactly once.
"""


def build_grading_prompt(
    messages: list[dict[str, str]],
    answer: str,
    criteria_texts: list[str],
) -> str:
    """Render the grading prompt for a single task.

    Args:
        messages: The task conversation as ``{"role": ..., "content": ...}`` dicts, in
            order. This is the conversation the candidate model answered.
        answer: The candidate answer being graded.
        criteria_texts: Criterion text only, in the order the judge should see it (the
            caller shuffles). Ordinals in the judge's reply are 1-based positions into
            this list. Never pass point values or the reference answer.

    Returns:
        The complete prompt for one judge call covering every criterion.
    """
    if not criteria_texts:
        raise ValueError("criteria_texts must contain at least one criterion")
    for index, message in enumerate(messages):
        if "role" not in message or "content" not in message:
            raise ValueError(
                f"messages[{index}] must have 'role' and 'content' keys, "
                f"got {sorted(message)!r}"
            )
    conversation = "\n\n".join(
        f"[{message['role']}]\n{message['content']}" for message in messages
    )
    criteria = "\n".join(f"{i}. {text}" for i, text in enumerate(criteria_texts, start=1))
    return GRADE_PROMPT.format(
        conversation=conversation,
        candidate_answer=answer,
        criteria=criteria,
        n=len(criteria_texts),
    )
