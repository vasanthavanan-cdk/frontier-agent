"""Premium node: calls Claude after the human has explicitly approved escalation.

This node only runs when `state.escalation_approved is True`. It passes the
original request plus all local attempt outputs as context so Claude can build
on the local work rather than starting from scratch.
"""
from __future__ import annotations

import os

from rich.console import Console
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from ..state import AgentOutput, AgentState
from ...config import settings
from ...logger import get_logger

console = Console()
log = get_logger(__name__)

_SYSTEM = """You are an expert AI assistant. A local AI system attempted this task but had low confidence.
Provide a high-quality, complete response."""


def _best_local(state: AgentState) -> str:
    """Return the highest-confidence local output, used as a graceful fallback."""
    outputs = [o for r, o in state.agent_outputs.items() if r != "premium"]
    if not outputs:
        return ""
    return max(outputs, key=lambda o: o.confidence).content


def premium_node(state: AgentState) -> AgentState:
    """Call Claude after human approval, degrading gracefully if it can't be reached."""
    console.print(f"[bold magenta]→ Escalating to {state.fallback_premium_model}...[/bold magenta]")

    # Premium requires an Anthropic API key — without it, escalation is impossible.
    # Fall back to the best local output rather than crashing.
    if not (settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")):
        console.print(
            "[yellow]⚠ ANTHROPIC_API_KEY not set — cannot escalate. "
            "Returning the best local result instead.[/yellow]\n"
            "[dim]  Set ANTHROPIC_API_KEY in your .env to enable premium escalation.[/dim]"
        )
        log.warning("[%s] PREMIUM skipped — no ANTHROPIC_API_KEY", state.task_id)
        return AgentState(**{
            **state.model_dump(),
            "current_step": "done",
            "final_output": _best_local(state),
        })

    # build context from local attempts
    local_context = ""
    for role, out in state.agent_outputs.items():
        local_context += f"\n--- Local {role} output (confidence: {out.confidence:.2f}) ---\n{out.content}\n"

    user_msg = (
        f"Original request: {state.original_input}\n\n"
        f"Local model attempts (for context):\n{local_context}"
    )

    try:
        llm = ChatAnthropic(model=state.fallback_premium_model, max_tokens=4096)
        response = llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=user_msg),
        ])
        content = response.content
    except Exception as e:
        console.print(f"[yellow]⚠ Premium call failed ({e}) — returning best local result.[/yellow]")
        log.warning("[%s] PREMIUM call failed: %s", state.task_id, e)
        return AgentState(**{
            **state.model_dump(),
            "current_step": "done",
            "final_output": _best_local(state),
        })

    # track premium token usage
    token_count = len(content) // 4
    cost = token_count * (3 / 1_000_000)

    output = AgentOutput(agent="premium", content=content, confidence=1.0)

    return AgentState(**{
        **state.model_dump(),
        "current_step": "done",
        "final_output": content,
        "agent_outputs": {**state.agent_outputs, "premium": output},
        "token_usage": state.token_usage.model_copy(update={
            "premium_tokens": state.token_usage.premium_tokens + token_count,
            "premium_cost_usd": state.token_usage.premium_cost_usd + cost,
        }),
    })
