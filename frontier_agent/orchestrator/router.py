from __future__ import annotations

from .state import AgentState


def after_planner(state: AgentState) -> str:
    threshold = state.confidence_thresholds.get("planner", state.confidence_thresholds["default"])
    score = state.confidence_scores.get("planner", 0.0)
    retries = state.retry_counts.get("planner", 0)

    if score < threshold and retries < state.max_local_retries:
        return "planner"  # retry
    if score < threshold and retries >= state.max_local_retries:
        return "escalate"
    return "coder"


def after_coder(state: AgentState) -> str:
    threshold = state.confidence_thresholds.get("coder", state.confidence_thresholds["default"])
    score = state.confidence_scores.get("coder", 0.0)
    retries = state.retry_counts.get("coder", 0)

    if score < threshold and retries < state.max_local_retries:
        return "coder"  # retry
    if score < threshold and retries >= state.max_local_retries:
        return "escalate"
    return "reviewer"


def after_reviewer(state: AgentState) -> str:
    threshold = state.confidence_thresholds.get("reviewer", state.confidence_thresholds["default"])
    score = state.confidence_scores.get("reviewer", 0.0)
    retries = state.retry_counts.get("reviewer", 0)

    if score < threshold and retries >= state.max_local_retries:
        return "escalate"
    return "done"


def after_escalation(state: AgentState) -> str:
    return "premium" if state.escalation_approved else "done"
