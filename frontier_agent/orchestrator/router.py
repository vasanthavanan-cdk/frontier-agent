"""Conditional edge functions for the LangGraph pipeline.

Each function receives the current AgentState and returns the name of the next
node to visit. Returning the same node name causes a retry; returning "escalate"
hands off to the human-approval gate.
"""
from __future__ import annotations

from .state import AgentState


def after_planner(state: AgentState) -> str:
    """Route after the planner: retry if confidence low, escalate after max retries, else proceed to coder."""
    threshold = state.confidence_thresholds.get("planner", state.confidence_thresholds["default"])
    score = state.confidence_scores.get("planner", 0.0)
    retries = state.retry_counts.get("planner", 0)

    if score < threshold and retries < state.max_local_retries:
        return "planner"  # retry
    if score < threshold and retries >= state.max_local_retries:
        return "escalate"
    return "coder"


def after_coder(state: AgentState) -> str:
    """Route after the coder: retry, escalate, or pass to reviewer."""
    threshold = state.confidence_thresholds.get("coder", state.confidence_thresholds["default"])
    score = state.confidence_scores.get("coder", 0.0)
    retries = state.retry_counts.get("coder", 0)

    if score < threshold and retries < state.max_local_retries:
        return "coder"  # retry
    if score < threshold and retries >= state.max_local_retries:
        return "escalate"
    return "reviewer"


def after_reviewer(state: AgentState) -> str:
    """Route after the reviewer: escalate if confidence still low, otherwise finish."""
    threshold = state.confidence_thresholds.get("reviewer", state.confidence_thresholds["default"])
    score = state.confidence_scores.get("reviewer", 0.0)
    retries = state.retry_counts.get("reviewer", 0)

    if score < threshold and retries >= state.max_local_retries:
        return "escalate"
    return "done"


def after_escalation(state: AgentState) -> str:
    """Route after the escalation gate: go to premium if approved, else finish with local output."""
    return "premium" if state.escalation_approved else "done"
