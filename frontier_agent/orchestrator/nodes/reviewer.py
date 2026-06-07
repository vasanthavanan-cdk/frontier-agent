from __future__ import annotations

from rich.console import Console

from ..state import AgentState
from .base import call_local_model

console = Console()

_SYSTEM = """You are a senior code reviewer. Review the implementation for:
- Correctness (does it solve the original request?)
- Edge cases and error handling
- Security issues (injection, unvalidated input, etc.)
- Performance concerns

Respond with:
VERDICT: APPROVED | NEEDS_REVISION
ISSUES:
- <issue 1>
- <issue 2> (or "None" if no issues)
SUMMARY: <one sentence overall assessment>"""


def reviewer_node(state: AgentState) -> AgentState:
    console.print("[dim]→ Reviewer checking output...[/dim]")

    coder_out = state.agent_outputs.get("coder")
    user_msg = (
        f"Original request: {state.original_input}\n\n"
        f"Implementation to review:\n{coder_out.content if coder_out else 'No implementation provided'}"
    )

    output, confidence = call_local_model(
        state=state,
        role="reviewer",
        system_prompt=_SYSTEM,
        user_message=user_msg,
    )

    return AgentState(**{
        **state.model_dump(),
        "current_step": "done",
        "agent_outputs": {**state.agent_outputs, "reviewer": output},
        "confidence_scores": {**state.confidence_scores, "reviewer": confidence},
        "retry_counts": {**state.retry_counts, "reviewer": state.retry_counts.get("reviewer", 0) + 1},
    })
