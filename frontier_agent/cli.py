from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
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
_STATE_DIR = Path.home() / ".frontier"
_OLLAMA_PID_FILE = _STATE_DIR / "ollama.pid"
_OLLAMA_URL = "http://localhost:11434"
_DEFAULT_MODEL = "gemma4:12b"


def _ollama_running() -> bool:
    try:
        urllib.request.urlopen(_OLLAMA_URL, timeout=2)
        return True
    except Exception:
        return False


def _wait_for_ollama(timeout: int = 20) -> bool:
    for _ in range(timeout):
        if _ollama_running():
            return True
        time.sleep(1)
    return False


# ── frontier start ───────────────────────────────────────────────────────────

@app.command("start")
def start() -> None:
    """Start Ollama and load the default model. Safe to call if already running."""
    import shutil

    console.print()

    # ── Ollama binary ─────────────────────────────────────────────────────────
    if not shutil.which("ollama"):
        console.print(Panel.fit(
            "[red bold]Ollama not found.[/red bold]\n"
            "Install the official app from [link]https://ollama.com/download/mac[/link]\n"
            "[dim](Do not use Homebrew — it ships the MLX-only backend.)[/dim]",
            border_style="red",
        ))
        raise typer.Exit(1)

    # ── Start Ollama if not running ───────────────────────────────────────────
    if _ollama_running():
        console.print("[green]✓[/green] Ollama already running")
    else:
        console.print("[dim]Starting Ollama...[/dim]")
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["OLLAMA_FLASH_ATTENTION"] = "1"
        env["OLLAMA_KV_CACHE_TYPE"] = "q8_0"
        log_path = _STATE_DIR / "ollama.log"
        with open(log_path, "w") as log:
            proc = subprocess.Popen(
                ["ollama", "serve"],
                env=env,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        _OLLAMA_PID_FILE.write_text(str(proc.pid))

        if _wait_for_ollama(timeout=20):
            console.print(f"[green]✓[/green] Ollama started  (pid {proc.pid})")
        else:
            console.print(
                "[red]✗ Ollama did not respond after 20 s.[/red]\n"
                f"  Check log: {log_path}"
            )
            raise typer.Exit(1)

    # ── Ensure default model is present ──────────────────────────────────────
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if _DEFAULT_MODEL in result.stdout:
        console.print(f"[green]✓[/green] {_DEFAULT_MODEL} ready")
    else:
        console.print(f"[dim]Pulling {_DEFAULT_MODEL} (~7.6 GB, one-time download)...[/dim]")
        pull = subprocess.run(["ollama", "pull", _DEFAULT_MODEL])
        if pull.returncode == 0:
            console.print(f"[green]✓[/green] {_DEFAULT_MODEL} downloaded")
        else:
            console.print(f"[red]✗ Failed to pull {_DEFAULT_MODEL}[/red]")
            raise typer.Exit(1)

    console.print()
    console.print(Panel.fit(
        "[green bold]Frontier Agent is ready.[/green bold]\n"
        "Run [bold]frontier run \"your task\"[/bold] or connect via MCP.",
        border_style="green",
    ))
    console.print()


# ── frontier stop ─────────────────────────────────────────────────────────────

@app.command("stop")
def stop() -> None:
    """Stop Ollama. The MCP server exits automatically with your agent session."""
    import shutil
    import signal

    console.print()

    if not _ollama_running():
        console.print("[dim]Ollama is not running — nothing to stop.[/dim]")
        console.print()
        return

    stopped = False

    # try saved PID first
    if _OLLAMA_PID_FILE.exists():
        try:
            pid = int(_OLLAMA_PID_FILE.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            _OLLAMA_PID_FILE.unlink(missing_ok=True)
            stopped = True
            console.print(f"[green]✓[/green] Sent SIGTERM to Ollama (pid {pid})")
        except (ProcessLookupError, ValueError):
            _OLLAMA_PID_FILE.unlink(missing_ok=True)

    # fallback: pkill
    if not stopped:
        r = subprocess.run(["pkill", "-TERM", "-x", "ollama"], capture_output=True)
        if r.returncode == 0:
            console.print("[green]✓[/green] Stopped Ollama via pkill")
        else:
            console.print("[yellow]⚠ Could not find an Ollama process to stop.[/yellow]")

    # wait up to 5 s for it to exit
    for _ in range(5):
        time.sleep(1)
        if not _ollama_running():
            break

    if _ollama_running():
        console.print("[yellow]⚠ Ollama still responding — may need a moment to shut down.[/yellow]")
    else:
        console.print("[green]✓[/green] Ollama stopped")

    console.print(
        "\n[dim]MCP server processes exit automatically when your agent session closes.[/dim]\n"
    )


# ── frontier install-global ───────────────────────────────────────────────────

@app.command("install-global")
def install_global(
    link_dir: str = typer.Option(
        "/usr/local/bin",
        "--dir",
        help="Directory to place the symlink (must be on PATH).",
    ),
) -> None:
    """Symlink the frontier binary into /usr/local/bin so it works from anywhere."""
    frontier_bin = Path(sys.executable).parent / "frontier"
    if not frontier_bin.exists():
        console.print(f"[red]frontier binary not found at {frontier_bin}[/red]")
        raise typer.Exit(1)

    target = Path(link_dir) / "frontier"
    if target.exists() or target.is_symlink():
        target.unlink()

    try:
        target.symlink_to(frontier_bin)
        console.print(f"[green]✓[/green] Symlink created: {target} → {frontier_bin}")
        console.print(
            "\n[dim]You can now run [bold]frontier[/bold] from any directory "
            "without activating the venv.[/dim]\n"
        )
    except PermissionError:
        console.print(
            f"[red]Permission denied writing to {link_dir}.[/red]\n"
            f"  Run with sudo:\n"
            f"  [bold]sudo {sys.executable.replace('python', 'frontier')} install-global[/bold]\n"
            f"  Or pick a dir you own:\n"
            f"  [bold]frontier install-global --dir ~/.local/bin[/bold]"
        )
        raise typer.Exit(1)


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


# ── frontier mcp-status ───────────────────────────────────────────────────────

@app.command("mcp-status")
def mcp_status() -> None:
    """Check whether the MCP server and Ollama are healthy."""
    import subprocess
    import urllib.request
    import sys

    console.print()
    all_ok = True

    # ── 1. Python executable reachable ───────────────────────────────────────
    python_bin = sys.executable
    console.print(f"[bold]Python[/bold]  {python_bin}")

    # ── 2. mcp package installed ─────────────────────────────────────────────
    try:
        import mcp  # noqa: F401
        import importlib.metadata
        mcp_ver = importlib.metadata.version("mcp")
        console.print(f"[bold]mcp pkg[/bold] [green]✓[/green]  v{mcp_ver}")
    except Exception:
        console.print("[bold]mcp pkg[/bold] [red]✗ not installed[/red] — run: pip install mcp")
        all_ok = False

    # ── 3. mcp_server module importable ──────────────────────────────────────
    try:
        from frontier_agent import mcp_server  # noqa: F401
        console.print("[bold]server [/bold] [green]✓[/green]  frontier_agent.mcp_server importable")
    except Exception as e:
        console.print(f"[bold]server [/bold] [red]✗ import failed:[/red] {e}")
        all_ok = False

    # ── 4. Ollama process reachable ───────────────────────────────────────────
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        console.print("[bold]Ollama [/bold] [green]✓[/green]  http://localhost:11434 responding")
    except Exception:
        console.print(
            "[bold]Ollama [/bold] [red]✗ not reachable[/red] — run: "
            "OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve"
        )
        all_ok = False

    # ── 5. gemma4:12b downloaded ──────────────────────────────────────────────
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if "gemma4:12b" in result.stdout:
        console.print("[bold]Model  [/bold] [green]✓[/green]  gemma4:12b present")
    else:
        console.print("[bold]Model  [/bold] [yellow]⚠ gemma4:12b not found[/yellow] — run: ollama pull gemma4:12b")
        all_ok = False

    # ── 6. claude mcp list (only if claude CLI available) ────────────────────
    import shutil
    if shutil.which("claude"):
        r = subprocess.run(["claude", "mcp", "list"], capture_output=True, text=True, timeout=10)
        if "frontier-agent" in r.stdout and "Connected" in r.stdout:
            console.print("[bold]Claude [/bold] [green]✓[/green]  frontier-agent registered and connected")
        elif "frontier-agent" in r.stdout:
            console.print("[bold]Claude [/bold] [yellow]⚠ frontier-agent registered but not connected[/yellow]")
            all_ok = False
        else:
            console.print(
                "[bold]Claude [/bold] [yellow]⚠ not registered[/yellow] — run:\n"
                "  claude mcp add --scope user frontier-agent -- "
                f"{sys.executable} -m frontier_agent.mcp_server"
            )
            all_ok = False
    else:
        console.print("[bold]Claude [/bold] [dim]– claude CLI not found (skip)[/dim]")

    # ── summary ───────────────────────────────────────────────────────────────
    console.print()
    if all_ok:
        console.print(Panel.fit(
            "[green bold]All checks passed.[/green bold]  "
            "MCP server is ready — any connected agent can call frontier-agent tools.",
            border_style="green",
        ))
    else:
        console.print(Panel.fit(
            "[yellow bold]Some checks failed.[/yellow bold]  "
            "Fix the items marked [red]✗[/red] or [yellow]⚠[/yellow] above.",
            border_style="yellow",
        ))
        raise typer.Exit(1)

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
