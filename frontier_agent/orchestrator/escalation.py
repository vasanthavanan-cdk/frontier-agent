"""Human-in-the-loop escalation gate.

Before any premium model is called, `request_escalation` prints a Rich panel
showing per-step confidence scores vs thresholds and an estimated cost, then
blocks on a y/N terminal prompt. The returned state carries `escalation_approved`.

This gate cannot be bypassed — `require_human_approval: true` is enforced at
schema load time (see workflows/schema.py::EscalationConfig).
"""
from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Confirm
from rich import box

from .state import AgentState

console = Console()

# claude-sonnet-4-5 pricing: $3/M input tokens, $15/M output tokens
_INPUT_COST_PER_TOKEN = 3 / 1_000_000
_OUTPUT_COST_PER_TOKEN = 15 / 1_000_000
_CHARS_PER_TOKEN = 4


def _estimate_tokens(state: AgentState) -> tuple[int, int]:
    """Estimate (input_tokens, output_tokens) from accumulated state content."""
    input_chars = len(state.original_input)
    for out in state.agent_outputs.values():
        input_chars += len(out.content)
    input_tokens = input_chars // _CHARS_PER_TOKEN
    output_tokens = input_tokens // 2  # rough estimate: output ≈ half input
    return input_tokens, output_tokens


def _confidence_bar(score: float, threshold: float, width: int = 20) -> str:
    """Render a Rich-formatted block-character progress bar coloured by pass/fail."""
    filled = int(score * width)
    bar = "█" * filled + "░" * (width - filled)
    color = "green" if score >= threshold else "red"
    return f"[{color}]{bar}[/{color}] {score:.2f}"


def request_escalation(state: AgentState) -> AgentState:
    """Block and ask the user for explicit approval before touching a premium model."""
    input_tokens, output_tokens = _estimate_tokens(state)
    estimated_cost = (
        input_tokens * _INPUT_COST_PER_TOKEN
        + output_tokens * _OUTPUT_COST_PER_TOKEN
    )

    # build confidence breakdown table
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    table.add_column("Step", style="white")
    table.add_column("Score", min_width=28)
    table.add_column("Threshold", justify="right")
    table.add_column("Status", justify="center")

    for step, score in state.confidence_scores.items():
        threshold = state.confidence_thresholds.get(step, state.confidence_thresholds["default"])
        status = "[green]OK[/green]" if score >= threshold else "[red]LOW[/red]"
        table.add_row(
            step,
            _confidence_bar(score, threshold),
            f"{threshold:.2f}",
            status,
        )

    console.print()
    console.print(Panel(
        table,
        title="[bold yellow]⚠  ESCALATION REQUESTED[/bold yellow]",
        border_style="yellow",
        padding=(1, 2),
    ))

    # cost breakdown
    cost_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    cost_table.add_column("Label", style="dim")
    cost_table.add_column("Value", style="white")
    cost_table.add_row("Local model", state.model_assignments.get("coder", "gemma4:12b"))
    cost_table.add_row("Escalation target", f"[magenta]{state.fallback_premium_model}[/magenta]")
    cost_table.add_row("Reason", state.escalation_reason or "Confidence below threshold after max retries")
    cost_table.add_row("Estimated input tokens", f"~{input_tokens:,}")
    cost_table.add_row("Estimated output tokens", f"~{output_tokens:,}")
    cost_table.add_row("Estimated cost", f"[bold]~${estimated_cost:.4f} USD[/bold]")

    console.print(Panel(
        cost_table,
        title="[bold]Cost Estimate[/bold]",
        border_style="dim",
        padding=(0, 2),
    ))
    console.print()

    approved = Confirm.ask(
        "[bold yellow]Approve escalation to premium model?[/bold yellow]",
        default=False,
    )

    if not approved:
        console.print("[dim]Escalation declined — returning best local output.[/dim]")

    return AgentState(
        **{**state.model_dump(), "escalation_approved": approved}
    )
