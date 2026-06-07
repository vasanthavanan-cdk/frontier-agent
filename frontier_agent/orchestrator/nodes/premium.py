from __future__ import annotations

from rich.console import Console
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage

from ..state import AgentOutput, AgentState

console = Console()

_SYSTEM = """You are an expert AI assistant. A local AI system attempted this task but had low confidence.
Provide a high-quality, complete response."""


def premium_node(state: AgentState) -> AgentState:
    """Call Claude after human approval."""
    console.print(f"[bold magenta]→ Escalating to {state.fallback_premium_model}...[/bold magenta]")

    # build context from local attempts
    local_context = ""
    for role, out in state.agent_outputs.items():
        local_context += f"\n--- Local {role} output (confidence: {out.confidence:.2f}) ---\n{out.content}\n"

    user_msg = (
        f"Original request: {state.original_input}\n\n"
        f"Local model attempts (for context):\n{local_context}"
    )

    llm = ChatAnthropic(model=state.fallback_premium_model, max_tokens=4096)
    response = llm.invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=user_msg),
    ])
    content = response.content

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
