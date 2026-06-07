from __future__ import annotations

import subprocess
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

app = typer.Typer(
    name="frontier",
    help="Frontier Agent — local-first AI orchestration with human-in-the-loop escalation.",
    no_args_is_help=True,
)
models_app = typer.Typer(help="Manage local models.")
app.add_typer(models_app, name="models")

console = Console()

_WORKFLOWS_DIR = Path(__file__).parent / "workflows" / "definitions"


# ── frontier run ─────────────────────────────────────────────────────────────

@app.command("run")
def run_task(
    task: str = typer.Argument(..., help="Task description in plain language."),
    workflow: str = typer.Option(
        "coding",
        "--workflow", "-w",
        help="Workflow to use: coding | research | review (or any custom workflow name).",
    ),
) -> None:
    """Run a task through the local agent pipeline."""
    from .orchestrator.graph import run
    run(task=task, workflow_name=workflow)


# ── frontier models ───────────────────────────────────────────────────────────

@models_app.command("status")
def models_status() -> None:
    """Show downloaded Ollama models with size and registry info."""
    from .models.registry import REGISTRY

    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        console.print("[red]Ollama is not running. Start it with: ollama serve[/red]")
        raise typer.Exit(1)

    lines = result.stdout.strip().splitlines()
    downloaded: dict[str, str] = {}
    for line in lines[1:]:   # skip header
        parts = line.split()
        if parts:
            tag = parts[0]
            size = parts[2] + " " + parts[3] if len(parts) > 3 else ""
            downloaded[tag] = size

    table = Table(
        title="Local Models",
        box=box.ROUNDED,
        header_style="bold cyan",
        show_lines=False,
    )
    table.add_column("Model", style="white")
    table.add_column("Downloaded", justify="center")
    table.add_column("Disk", justify="right")
    table.add_column("Roles", style="dim")
    table.add_column("Notes")

    for tag, info in REGISTRY.items():
        is_downloaded = tag in downloaded
        dl_marker = "[green]✓[/green]" if is_downloaded else "[dim]–[/dim]"
        size_str = downloaded.get(tag, f"~{info.size_gb:.0f} GB")
        roles = ", ".join(info.roles)
        table.add_row(tag, dl_marker, size_str, roles, info.description)

    console.print()
    console.print(table)

    total_downloaded = sum(
        info.size_gb for tag, info in REGISTRY.items() if tag in downloaded
    )
    console.print(f"\n[dim]Total downloaded registry models: ~{total_downloaded:.1f} GB[/dim]")
    console.print("[dim]Run [bold]frontier models pull <role>[/bold] to download a specialist.[/dim]\n")


@models_app.command("pull")
def models_pull(
    role: str = typer.Argument(
        ...,
        help="Specialist role to pull: coding | reasoning | orchestrator",
    ),
) -> None:
    """Pull a recommended specialist model for a given role."""
    from .models.registry import ROLE_RECOMMENDATIONS

    tag = ROLE_RECOMMENDATIONS.get(role)
    if not tag:
        console.print(
            f"[red]Unknown role '{role}'. Available: {', '.join(ROLE_RECOMMENDATIONS)}[/red]"
        )
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"Pulling [cyan]{tag}[/cyan] for role [bold]{role}[/bold]\n"
        f"[dim]This may take several minutes depending on your connection.[/dim]",
        border_style="cyan",
    ))
    subprocess.run(["ollama", "pull", tag], check=True)
    console.print(f"\n[green]✓ {tag} downloaded. Update your workflow YAML to use it.[/green]")


@models_app.command("list")
def models_list() -> None:
    """List all models currently downloaded in Ollama (raw)."""
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    console.print(result.stdout or result.stderr)


# ── frontier workflows ────────────────────────────────────────────────────────

@app.command("workflows")
def list_workflows() -> None:
    """List available workflow definitions."""
    from .workflows.loader import load_workflow_file

    yamls = sorted(_WORKFLOWS_DIR.glob("*.yaml"))
    if not yamls:
        console.print("[yellow]No workflow definitions found.[/yellow]")
        return

    table = Table(box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Name")
    table.add_column("Steps", justify="center")
    table.add_column("Coder model")
    table.add_column("Coder threshold", justify="right")
    table.add_column("Description")

    for path in yamls:
        try:
            w = load_workflow_file(path)
            table.add_row(
                w.name,
                str(len(w.steps)),
                w.models.coder,
                str(w.confidence_thresholds.coder),
                w.description,
            )
        except Exception as e:
            table.add_row(path.stem, "[red]ERROR[/red]", "", "", str(e))

    console.print()
    console.print(table)
    console.print(
        f"\n[dim]Workflow files are at: {_WORKFLOWS_DIR}[/dim]\n"
        "[dim]Edit them directly — changes take effect on next run.[/dim]\n"
    )


# ── frontier status ───────────────────────────────────────────────────────────

@app.command("status")
def status() -> None:
    """Show system status: Ollama, loaded models, available workflows."""
    import shutil

    console.print()

    # Ollama
    ollama_bin = shutil.which("ollama")
    if ollama_bin:
        ver = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        version_str = ver.stdout.strip().replace("ollama version is ", "")
        # check server
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434", timeout=2)
            server_status = "[green]running[/green]"
        except Exception:
            server_status = "[red]not running — start with: ollama serve[/red]"
        console.print(f"[bold]Ollama:[/bold] {version_str}  {server_status}")
    else:
        console.print("[bold]Ollama:[/bold] [red]not installed[/red]")

    # Workflows
    yamls = list(_WORKFLOWS_DIR.glob("*.yaml"))
    console.print(f"[bold]Workflows:[/bold] {len(yamls)} available ({', '.join(p.stem for p in yamls)})")

    # Models
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    lines = [l for l in result.stdout.strip().splitlines() if l and not l.startswith("NAME")]
    console.print(f"[bold]Downloaded models:[/bold] {len(lines)}")
    for line in lines:
        console.print(f"  [dim]{line.split()[0]}[/dim]")
    console.print()


# ── frontier bench ────────────────────────────────────────────────────────────

@app.command("bench")
def bench(
    category: str = typer.Option(
        "all",
        "--category", "-c",
        help="Task category to run: all | coding | reasoning | research",
    ),
    premium: bool = typer.Option(
        False,
        "--premium",
        help="Also run each task against Claude for comparison (requires ANTHROPIC_API_KEY).",
    ),
    model: str = typer.Option(
        "claude-sonnet-4-5",
        "--model",
        help="Premium model to use for comparison.",
    ),
) -> None:
    """Run the benchmark suite and print a report. Target: ≥70% local resolution at quality ≥4/5."""
    import os
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from benchmarks.runner import run_benchmark

    if premium and not os.getenv("ANTHROPIC_API_KEY"):
        console.print("[yellow]Warning: ANTHROPIC_API_KEY not set — skipping premium comparison.[/yellow]")
        premium = False

    cats = None if category == "all" else [category]
    report = run_benchmark(categories=cats, include_premium=premium, premium_model=model)
    report.print_summary()

    # write markdown report
    _write_md_report(report)


def _write_md_report(report: "BenchReport") -> None:
    from datetime import datetime
    from benchmarks.runner import BenchReport

    lines: list[str] = [
        "# Milestone 5 — Benchmark Report",
        f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Model**: gemma4:12b (local)  ",
        f"**Hardware**: Mac Mini M4 Pro 24GB  \n",
        "## Results by Task\n",
        "| ID | Category | Score | Escalated | Latency (s) | Cost (USD) | Judge Reason |",
        "|----|----------|-------|-----------|-------------|------------|--------------|",
    ]
    for r in report.local_results:
        esc = "Yes" if r.escalated else "No"
        lines.append(
            f"| {r.task_id} | {r.category} | {r.score}/5 | {esc} "
            f"| {r.latency_sec:.1f}s | ${r.premium_cost_usd:.4f} | {r.score_reason[:60]} |"
        )

    lr = report.local_results
    if lr:
        avg_score = sum(r.score for r in lr) / len(lr)
        n_escalated = sum(1 for r in lr if r.escalated)
        n_local = sum(1 for r in lr if not r.escalated and r.score >= 4)
        resolution_rate = n_local / len(lr)
        total_cost = sum(r.premium_cost_usd for r in lr)
        direct_est = len(lr) * ((800 * 3 / 1_000_000) + (600 * 15 / 1_000_000))

        lines += [
            "\n## Summary\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total tasks | {len(lr)} |",
            f"| Local resolution rate (score ≥4, no escalation) | {resolution_rate:.0%} |",
            f"| Average quality score | {avg_score:.1f}/5 |",
            f"| Escalations | {n_escalated} |",
            f"| Actual premium cost | ${total_cost:.4f} |",
            f"| Direct-Claude equivalent | ${direct_est:.4f} |",
            f"| Estimated savings | ${direct_est - total_cost:.4f} ({(direct_est - total_cost)/direct_est*100:.0f}%) |",
            f"\n**Target met**: {'Yes ✓' if resolution_rate >= 0.70 else 'No — below 70% threshold'}",
        ]

    out = Path(__file__).parent.parent / "reports" / "milestone-5-benchmark-report.md"
    out.write_text("\n".join(lines))
    console.print(f"[dim]Report written to {out}[/dim]\n")


if __name__ == "__main__":
    app()
