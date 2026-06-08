"""Coder node: generates a complete implementation based on the planner's output.

Receives the plan from `state.agent_outputs["planner"]` and produces fenced
code blocks. Enforces no-placeholder / no-TODO constraints via the system prompt;
these are later verified by the confidence scorer.
"""
from __future__ import annotations

from rich.console import Console

from ..state import AgentState
from .base import call_local_model

console = Console()

_SYSTEM = """You are an expert software engineer. Your job is to implement code based on a plan.

Rules:
- Write complete, runnable code — no placeholders or TODOs
- Use idiomatic patterns for the language
- Include type hints for Python code
- Wrap all code in fenced code blocks (```language ... ```)
- If the task is not a coding task, state that clearly and provide analysis instead"""


def coder_node(state: AgentState) -> AgentState:
    """LangGraph node: implement the plan and write the result into state."""
    console.print("[dim]→ Coder implementing...[/dim]")

    plan = state.agent_outputs.get("planner")
    user_msg = (
        f"Original request: {state.original_input}\n\n"
        f"Plan:\n{plan.content if plan else 'No plan available'}"
    )

    output, confidence = call_local_model(
        state=state,
        role="coder",
        system_prompt=_SYSTEM,
        user_message=user_msg,
    )

    return AgentState(**{
        **state.model_dump(),
        "current_step": "reviewer",
        "agent_outputs": {**state.agent_outputs, "coder": output},
        "confidence_scores": {**state.confidence_scores, "coder": confidence},
        "retry_counts": {**state.retry_counts, "coder": state.retry_counts.get("coder", 0) + 1},
    })
