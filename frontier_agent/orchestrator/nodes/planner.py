from __future__ import annotations

from rich.console import Console

from ..state import AgentState, TaskType
from .base import call_local_model

console = Console()

_SYSTEM = """You are a precise task planning agent. Your job is to:
1. Analyze the user's request and determine its type (coding, research, review, documentation, debugging, planning, general).
2. Break the task into clear, numbered subtasks with explicit dependencies.
3. Identify which specialist agent should handle each subtask.

Respond in this format:
TASK_TYPE: <one of: coding|research|review|documentation|debugging|planning|general>
PLAN:
1. <subtask> [agent: <planner|coder|reviewer|documenter>]
2. <subtask> (depends on 1) [agent: ...]
...
SUMMARY: <one sentence describing the overall goal>"""


def planner_node(state: AgentState) -> AgentState:
    console.print("[dim]→ Planner analyzing task...[/dim]")

    output, confidence = call_local_model(
        state=state,
        role="planner",
        system_prompt=_SYSTEM,
        user_message=state.original_input,
    )

    # detect task type from planner output
    task_type = state.task_type
    for line in output.content.splitlines():
        if line.startswith("TASK_TYPE:"):
            detected = line.split(":", 1)[1].strip().lower()
            try:
                task_type = TaskType(detected)
            except ValueError:
                pass
            break

    return AgentState(**{
        **state.model_dump(),
        "current_step": "coder",
        "task_type": task_type,
        "agent_outputs": {**state.agent_outputs, "planner": output},
        "confidence_scores": {**state.confidence_scores, "planner": confidence},
        "retry_counts": {**state.retry_counts, "planner": state.retry_counts.get("planner", 0) + 1},
    })
