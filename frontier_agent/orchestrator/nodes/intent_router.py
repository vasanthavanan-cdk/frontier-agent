"""Intent router node: auto-detects whether a task needs web search or local reasoning.

Runs as the graph entry point for the "auto" workflow. Sets tool_calling_enabled
and research_enabled on state so _after_researcher routes correctly. No-op for
explicit workflows (coding / research / review).

Routing logic:
- Web search (tool_agent path) is the DEFAULT for all tasks involving APIs,
  libraries, frameworks, tools, installations, or current facts — these are
  prone to stale training data.
- Local reasoning only for pure analytical/algorithmic tasks with no dependency
  on external state, library versions, or current information.
"""
from __future__ import annotations

import re

from rich.console import Console

from ..state import AgentState
from ...logger import get_logger

console = Console()
log = get_logger(__name__)

# Signals for pure reasoning/analytical tasks: no external dependency
_REASONING_PATTERNS = re.compile(
    r"\b(explain|what is|how does|why does|compare|difference between|"
    r"algorithm|big.?o|complexity|design pattern|cap theorem|solid principle|"
    r"fibonacci|factorial|sort|recursion|proof|mathematical|theoretical|"
    r"how do i think about|concept of|fundamentals of)\b",
    re.IGNORECASE,
)

# Override: even if query looks like reasoning, web is needed for these topics
_WEB_OVERRIDE_PATTERNS = re.compile(
    r"\b(latest|version|install|import|api|library|framework|package|module|"
    r"deprecated|upgrade|migrate|release|changelog|error|bug|fix|setup|"
    r"configure|config|update|new feature|breaking change|current|today|"
    r"how to use|tutorial|example|snippet|syntax)\b",
    re.IGNORECASE,
)


def intent_router_node(state: AgentState) -> AgentState:
    """Route to tool_agent (web) or planner (local) based on task content."""
    if state.workflow_name != "auto":
        log.debug("[%s] INTENT_ROUTER bypass (explicit workflow=%s)", state.task_id, state.workflow_name)
        return state

    text = state.original_input
    is_pure_reasoning = (
        bool(_REASONING_PATTERNS.search(text))
        and not bool(_WEB_OVERRIDE_PATTERNS.search(text))
    )
    use_web = not is_pure_reasoning

    route_label = "tool_agent (web)" if use_web else "planner (local reasoning)"
    log.info("[%s] INTENT_ROUTER → %s  pure_reasoning=%s  task=%s",
             state.task_id, route_label, is_pure_reasoning, text[:80])

    if state.interactive:
        icon = "🌐" if use_web else "🧠"
        console.print(f"[dim]→ Auto-routing: {icon} {route_label}[/dim]")

    return AgentState(**{
        **state.model_dump(),
        "tool_calling_enabled": use_web,
        "research_enabled": use_web,
    })
