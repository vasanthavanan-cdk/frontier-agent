"""LangGraph pipeline: builds and runs the multi-agent graph for a single task.

Graph topology: planner → coder → reviewer → (escalate?) → premium? → done
Each node writes to AgentState; conditional edges (router.py) decide the next hop.
Entry point for both the CLI and the MCP server is `run()`.
"""
from __future__ import annotations

import uuid
from rich.console import Console
from rich.panel import Panel

from langgraph.graph import StateGraph, END

from .state import AgentState, TokenUsage
from .nodes.planner import planner_node
from .nodes.coder import coder_node
from .nodes.reviewer import reviewer_node
from .nodes.premium import premium_node
from .escalation import request_escalation
from .router import after_planner, after_coder, after_reviewer, after_escalation
from ..workflows.loader import load_workflow, apply_workflow
from ..logger import get_logger

console = Console()
log = get_logger(__name__)


def _escalation_node(state: AgentState) -> AgentState:
    """Wrap request_escalation so the graph can treat it as a normal node."""
    log.warning("[%s] ESCALATION requested  reason=%s", state.task_id, state.escalation_reason or "low confidence")
    updated = request_escalation(state)
    approved = updated.escalation_approved
    log.warning("[%s] ESCALATION %s by user", state.task_id, "APPROVED" if approved else "DECLINED")
    return AgentState(**{
        **state.model_dump(),
        "escalation_requested": True,
        "escalation_approved": approved,
    })


def _done_node(state: AgentState) -> AgentState:
    """Finalise output: fall back to the coder's content if premium never ran."""
    # if final_output not set by premium, compile from local outputs
    if not state.final_output:
        coder_out = state.agent_outputs.get("coder")
        final = coder_out.content if coder_out else "No output generated."
        return AgentState(**{**state.model_dump(), "final_output": final})
    return state


def build_graph() -> StateGraph:
    """Compile and return the LangGraph StateGraph. Called once per `run()` invocation."""
    g = StateGraph(AgentState)

    g.add_node("planner", planner_node)
    g.add_node("coder", coder_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("escalate", _escalation_node)
    g.add_node("premium", premium_node)
    g.add_node("done", _done_node)

    g.set_entry_point("planner")

    g.add_conditional_edges("planner", after_planner, {
        "planner": "planner",
        "coder": "coder",
        "escalate": "escalate",
    })
    g.add_conditional_edges("coder", after_coder, {
        "coder": "coder",
        "reviewer": "reviewer",
        "escalate": "escalate",
    })
    g.add_conditional_edges("reviewer", after_reviewer, {
        "escalate": "escalate",
        "done": "done",
    })
    g.add_conditional_edges("escalate", after_escalation, {
        "premium": "premium",
        "done": "done",
    })
    g.add_edge("premium", "done")
    g.add_edge("done", END)

    return g.compile()


def run(task: str, workflow_name: str = "default_coding") -> AgentState:
    """Execute the full pipeline for `task` and return the final AgentState.

    Loads the named YAML workflow (falls back to defaults on missing file), seeds
    AgentState, runs the compiled graph, prints a Rich summary, and returns the
    completed state so callers (CLI, MCP server, benchmarks) can inspect results.
    """
    console.print(Panel.fit(
        f"[bold green]Task:[/bold green] {task}\n"
        f"[dim]Workflow: {workflow_name}[/dim]",
        title="[bold]Frontier Agent[/bold]",
        border_style="green",
    ))

    # load workflow config and overlay onto initial state
    try:
        workflow = load_workflow(workflow_name)
    except FileNotFoundError:
        console.print(f"[yellow]Warning: workflow '{workflow_name}' not found, using defaults.[/yellow]")
        workflow = None

    initial = AgentState(
        task_id=str(uuid.uuid4())[:8],
        original_input=task,
        workflow_name=workflow_name,
        token_usage=TokenUsage(),
    )
    if workflow:
        initial = apply_workflow(initial, workflow)

    log.info("[%s] PIPELINE START  workflow=%s  task=%s", initial.task_id, workflow_name, task[:80])

    graph = build_graph()
    result = graph.invoke(initial)
    final = AgentState(**result) if isinstance(result, dict) else result

    log.info(
        "[%s] PIPELINE DONE  escalated=%s  local_tokens=%d  premium_tokens=%d  cost=$%.4f",
        final.task_id,
        final.escalation_approved,
        final.token_usage.local_tokens,
        final.token_usage.premium_tokens,
        final.token_usage.premium_cost_usd,
    )

    _print_summary(final)
    return final


def _print_summary(state: AgentState) -> None:
    """Print a Rich panel showing token usage, cost, and escalation status."""
    usage = state.token_usage
    console.print()
    console.print(Panel.fit(
        f"[white]Local tokens:[/white] {usage.local_tokens:,}\n"
        f"[white]Premium tokens:[/white] {usage.premium_tokens:,}\n"
        f"[white]Premium cost:[/white] ${usage.premium_cost_usd:.4f} USD\n"
        f"[white]Escalated:[/white] {'Yes' if state.escalation_approved else 'No'}",
        title="[bold]Session Summary[/bold]",
        border_style="blue",
    ))
    console.print()
    console.print("[bold]Final Output:[/bold]")
    console.print(state.final_output)
