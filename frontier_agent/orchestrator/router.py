"""Conditional edge functions for the LangGraph pipeline.

Each function receives the current AgentState and returns the name of the next
node to visit. Returning the same node name causes a retry; returning "escalate"
hands off to the human-approval gate.
"""
from __future__ import annotations

from .state import AgentState
from ..logger import get_logger

log = get_logger(__name__)


def after_planner(state: AgentState) -> str:
    """Route after the planner: retry if confidence low, escalate after max retries, else proceed to coder."""
    threshold = state.confidence_thresholds.get("planner", state.confidence_thresholds["default"])
    score = state.confidence_scores.get("planner", 0.0)
    retries = state.retry_counts.get("planner", 0)

    if score < threshold and retries < state.max_local_retries:
        log.info("[%s] ROUTER  planner → RETRY (score=%.3f < %.2f, retry %d/%d)",
                 state.task_id, score, threshold, retries, state.max_local_retries)
        return "planner"
    if score < threshold and retries >= state.max_local_retries:
        log.warning("[%s] ROUTER  planner → ESCALATE (score=%.3f < %.2f after %d retries)",
                    state.task_id, score, threshold, retries)
        return "escalate"
    log.info("[%s] ROUTER  planner → coder (score=%.3f ≥ %.2f)", state.task_id, score, threshold)
    return "coder"


def after_coder(state: AgentState) -> str:
    """Route after the coder: retry, escalate, or pass to reviewer."""
    threshold = state.confidence_thresholds.get("coder", state.confidence_thresholds["default"])
    score = state.confidence_scores.get("coder", 0.0)
    retries = state.retry_counts.get("coder", 0)

    if score < threshold and retries < state.max_local_retries:
        log.info("[%s] ROUTER  coder → RETRY (score=%.3f < %.2f, retry %d/%d)",
                 state.task_id, score, threshold, retries, state.max_local_retries)
        return "coder"
    if score < threshold and retries >= state.max_local_retries:
        log.warning("[%s] ROUTER  coder → ESCALATE (score=%.3f < %.2f after %d retries)",
                    state.task_id, score, threshold, retries)
        return "escalate"
    log.info("[%s] ROUTER  coder → reviewer (score=%.3f ≥ %.2f)", state.task_id, score, threshold)
    return "reviewer"


def after_reviewer(state: AgentState) -> str:
    """Route after the reviewer: escalate if confidence still low, otherwise finish."""
    threshold = state.confidence_thresholds.get("reviewer", state.confidence_thresholds["default"])
    score = state.confidence_scores.get("reviewer", 0.0)
    retries = state.retry_counts.get("reviewer", 0)

    if score < threshold and retries >= state.max_local_retries:
        log.warning("[%s] ROUTER  reviewer → ESCALATE (score=%.3f < %.2f after %d retries)",
                    state.task_id, score, threshold, retries)
        return "escalate"
    log.info("[%s] ROUTER  reviewer → done (score=%.3f ≥ %.2f)", state.task_id, score, threshold)
    return "done"


def after_escalation(state: AgentState) -> str:
    """Route after the escalation gate: go to premium if approved, else finish with local output."""
    decision = "premium" if state.escalation_approved else "done (declined)"
    log.info("[%s] ROUTER  escalation → %s", state.task_id, decision)
    return "premium" if state.escalation_approved else "done"
