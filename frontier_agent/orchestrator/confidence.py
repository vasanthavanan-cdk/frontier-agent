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


def _check_completeness(output: AgentOutput, task_type: TaskType) -> float:
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


def compute_confidence(
    output: AgentOutput,
    task_type: TaskType,
    attempt: int = 1,
) -> float:
    completeness = _check_completeness(output, task_type)
    hedging_penalty = _check_hedging(output)
    structure = _validate_structure(output, task_type)

    base = (
        completeness * 0.40
        + (1.0 - hedging_penalty) * 0.25
        + structure * 0.35
    )
    retry_penalty = 0.05 * (attempt - 1)
    return max(0.0, round(base - retry_penalty, 3))
