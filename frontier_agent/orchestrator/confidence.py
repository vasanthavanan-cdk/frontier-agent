"""Behavioural confidence scoring for agent outputs.

Scores are derived from observable signals (output completeness, hedging language,
structural validity) rather than model self-reported probabilities, which are
unreliable for instruction-tuned models.

Public API: `compute_confidence(output, task_type, attempt) → float [0.0, 1.0]`
"""
from __future__ import annotations

import re

from .state import AgentOutput, TaskType


_HEDGING_PATTERNS = re.compile(
    r"\b(i think|i believe|i'm not sure|not sure|probably|might be|could be|"
    r"possibly|i'm uncertain|i don't know|unclear|i may be wrong|perhaps|"
    r"i would guess|hard to say|it depends|i cannot guarantee)\b",
    re.IGNORECASE,
)

_CODE_BLOCK_PATTERN = re.compile(r"```[\w]*\n.+?```", re.DOTALL)
_TODO_PATTERN = re.compile(r"\b(TODO|FIXME|PLACEHOLDER|pass\s*#|\.\.\.)\b")
_FUNCTION_PATTERN = re.compile(r"\bdef \w+\(|class \w+[:(]")

# Specific, falsifiable factual tokens: version-like numbers (1.10.0, v5.1.8)
# and 4-digit years. These are the claims a model is most likely to hallucinate
# in a research answer, and the easiest to verify against retrieved sources.
_CLAIM_PATTERN = re.compile(r"\bv?\d+\.\d+(?:\.\d+)*\b|\b(?:19|20)\d{2}\b")


def _check_completeness(output: AgentOutput, task_type: TaskType) -> float:
    """Score 0-1 for whether the output looks substantively complete for its task type."""
    text = output.content.strip()
    if not text:
        return 0.0
    if task_type == TaskType.coding:
        has_code = bool(_CODE_BLOCK_PATTERN.search(text))
        has_structure = bool(_FUNCTION_PATTERN.search(text))
        has_todos = bool(_TODO_PATTERN.search(text))
        score = 0.6 if has_code else 0.2
        score += 0.3 if has_structure else 0.0
        score -= 0.3 if has_todos else 0.0
        return max(0.0, min(1.0, score))
    if task_type == TaskType.planning:
        has_steps = bool(re.search(r"(\d+\.|[-*•])\s+\w", text))
        has_deps = bool(re.search(r"depends on|after step|requires", text, re.IGNORECASE))
        return 0.6 + (0.2 if has_steps else 0.0) + (0.2 if has_deps else 0.0)
    # general: length heuristic
    word_count = len(text.split())
    return min(1.0, word_count / 100)


def _check_hedging(output: AgentOutput) -> float:
    """Return fraction of hedging phrases — lower is better confidence."""
    matches = _HEDGING_PATTERNS.findall(output.content)
    # normalise: 0 matches → 0.0 penalty, 3+ matches → 1.0 penalty
    return min(1.0, len(matches) / 3)


def _validate_structure(output: AgentOutput, task_type: TaskType) -> float:
    """Score 0-1 for structural validity: balanced syntax for code, headers for docs."""
    if task_type == TaskType.coding:
        # try to parse code blocks
        blocks = _CODE_BLOCK_PATTERN.findall(output.content)
        if not blocks:
            return 0.3
        # basic syntax check: balanced braces/parens
        code = "\n".join(blocks)
        if code.count("(") - code.count(")") > 5:
            return 0.5
        return 1.0
    if task_type in (TaskType.research, TaskType.documentation):
        has_sections = bool(re.search(r"^#{1,3} .+", output.content, re.MULTILINE))
        return 0.8 if has_sections else 0.6
    return 0.8  # default: assume valid for non-structured types


def _check_grounding(output: AgentOutput, context: str) -> float | None:
    """Verify the output's specific factual claims against retrieved source context.

    Returns the fraction of version/year claims in the output that also appear in
    `context`, or None when the output makes no checkable factual claims (so the
    grounding gate stays out of the way for code, plans, and prose).

    This is the signal that catches *confident hallucination*: an answer that
    asserts "shaka-player v1.10.0" when the sources only mention v5.1.8 scores
    near 0 here regardless of how well-structured and unhedged it reads.
    """
    claims = set(_CLAIM_PATTERN.findall(output.content))
    if not claims:
        return None
    ctx = context.lower()
    supported = sum(1 for c in claims if c.lower() in ctx)
    return supported / len(claims)


def compute_confidence(
    output: AgentOutput,
    task_type: TaskType,
    attempt: int = 1,
    grounding_context: str | None = None,
) -> float:
    """Return a composite confidence score in [0.0, 1.0].

    Base weights: completeness 40%, inverse-hedging 25%, structure 35%.

    When `grounding_context` (retrieved source material) is supplied and the
    output makes specific, falsifiable claims (version numbers, years), a
    grounding signal is blended in: unsupported claims pull the score down, and
    if the majority of claims are unsupported the score is hard-capped at the
    grounding fraction — forcing escalation rather than returning a confident
    but ungrounded answer.

    Each retry attempt subtracts 0.05 to reflect diminishing returns.
    """
    completeness = _check_completeness(output, task_type)
    hedging_penalty = _check_hedging(output)
    structure = _validate_structure(output, task_type)

    base = (
        completeness * 0.40
        + (1.0 - hedging_penalty) * 0.25
        + structure * 0.35
    )

    if grounding_context:
        grounding = _check_grounding(output, grounding_context)
        if grounding is not None:
            base = 0.6 * base + 0.4 * grounding
            if grounding < 0.5:
                # majority of factual claims unsupported by sources — don't trust it
                base = min(base, grounding)

    retry_penalty = 0.05 * (attempt - 1)
    return max(0.0, round(base - retry_penalty, 3))
