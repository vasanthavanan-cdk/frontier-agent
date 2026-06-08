"""Benchmark runner: executes tasks from the suite against local and/or premium backends.

Main entry point: `run_benchmark(categories, include_premium, premium_model) → BenchReport`
Results are saved to benchmarks/results/latest.json and returned for further processing.

BenchReport computes derived metrics (local resolution rate, cost savings) as properties
so they stay consistent with the raw results at all times.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import box

from .tasks.suite import TASKS, BenchTask, Category
from .judge import judge

console = Console()
_RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class TaskResult:
    """Result for a single task run against one backend (local or premium)."""
    task_id: str
    category: str
    prompt: str
    backend: str              # "local" or "premium"
    output: str
    score: int                # 1-5
    score_reason: str
    escalated: bool
    local_tokens: int
    premium_tokens: int
    premium_cost_usd: float
    latency_sec: float
    error: str = ""


@dataclass
class BenchReport:
    """Aggregated results for a full benchmark run with computed summary metrics."""
    results: list[TaskResult] = field(default_factory=list)

    # ── computed metrics ──────────────────────────────────────────────────
    @property
    def local_results(self) -> list[TaskResult]:
        return [r for r in self.results if r.backend == "local"]

    @property
    def premium_results(self) -> list[TaskResult]:
        return [r for r in self.results if r.backend == "premium"]

    def _avg_score(self, results: list[TaskResult]) -> float:
        return sum(r.score for r in results) / max(len(results), 1)

    def _local_resolution_rate(self) -> float:
        lr = self.local_results
        resolved = sum(1 for r in lr if not r.escalated and r.score >= 4)
        return resolved / max(len(lr), 1)

    def _total_premium_cost(self) -> float:
        return sum(r.premium_cost_usd for r in self.results)

    def _direct_cost_estimate(self) -> float:
        """What it would have cost to run everything through Claude directly."""
        # estimate: each task ~800 input tokens + ~600 output tokens → $3/M + $15/M
        n = len(self.local_results)
        return n * ((800 * 3 / 1_000_000) + (600 * 15 / 1_000_000))

    def print_summary(self) -> None:
        lr = self.local_results
        pr = self.premium_results

        # per-category breakdown
        table = Table(title="Benchmark Results", box=box.ROUNDED, header_style="bold cyan")
        table.add_column("Category", style="white")
        table.add_column("Tasks", justify="center")
        table.add_column("Local avg score", justify="center")
        table.add_column("Escalated", justify="center")
        table.add_column("Local ≥4 rate", justify="center")

        for cat in Category:
            cat_results = [r for r in lr if r.category == cat.value]
            if not cat_results:
                continue
            avg = sum(r.score for r in cat_results) / len(cat_results)
            n_escalated = sum(1 for r in cat_results if r.escalated)
            rate = sum(1 for r in cat_results if not r.escalated and r.score >= 4)
            table.add_row(
                cat.value,
                str(len(cat_results)),
                f"{avg:.1f}/5",
                f"{n_escalated}/{len(cat_results)}",
                f"{rate/len(cat_results):.0%}",
            )

        console.print()
        console.print(table)

        local_res_rate = self._local_resolution_rate()
        actual_cost = self._total_premium_cost()
        direct_cost = self._direct_cost_estimate()
        savings = direct_cost - actual_cost
        savings_pct = (savings / direct_cost * 100) if direct_cost > 0 else 0

        console.print()
        console.print(f"  [bold]Overall local resolution rate:[/bold]  {local_res_rate:.0%}  "
                      f"({'[green]TARGET MET[/green]' if local_res_rate >= 0.70 else '[yellow]BELOW 70% TARGET[/yellow]'})")
        console.print(f"  [bold]Avg quality (local):[/bold]            {self._avg_score(lr):.1f}/5")
        if pr:
            console.print(f"  [bold]Avg quality (premium):[/bold]         {self._avg_score(pr):.1f}/5")
        console.print(f"  [bold]Actual premium cost:[/bold]            ${actual_cost:.4f} USD")
        console.print(f"  [bold]Direct-Claude equivalent cost:[/bold]  ${direct_cost:.4f} USD")
        console.print(f"  [bold]Estimated savings:[/bold]              ${savings:.4f} USD ({savings_pct:.0f}%)")
        console.print()


def _run_local(task: BenchTask) -> TaskResult:
    """Run one task through the Frontier Agent pipeline and score it with the judge."""
    from frontier_agent.orchestrator.graph import run as fa_run

    start = time.perf_counter()
    try:
        state = fa_run(task=task.prompt, workflow_name=task.workflow)
        latency = time.perf_counter() - start
        score, reason = judge(task, state.final_output)
        return TaskResult(
            task_id=task.id,
            category=task.category.value,
            prompt=task.prompt,
            backend="local",
            output=state.final_output,
            score=score,
            score_reason=reason,
            escalated=state.escalation_approved,
            local_tokens=state.token_usage.local_tokens,
            premium_tokens=state.token_usage.premium_tokens,
            premium_cost_usd=state.token_usage.premium_cost_usd,
            latency_sec=round(latency, 2),
        )
    except Exception as e:
        return TaskResult(
            task_id=task.id, category=task.category.value, prompt=task.prompt,
            backend="local", output="", score=1, score_reason="Error",
            escalated=False, local_tokens=0, premium_tokens=0,
            premium_cost_usd=0.0, latency_sec=0.0, error=str(e),
        )


def _run_premium(task: BenchTask, model: str = "claude-sonnet-4-5") -> TaskResult:
    """Run one task directly against a Claude model for comparison scoring."""
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage

    start = time.perf_counter()
    try:
        llm = ChatAnthropic(model=model, max_tokens=2048)
        response = llm.invoke([HumanMessage(content=task.prompt)])
        latency = time.perf_counter() - start
        output = response.content
        score, reason = judge(task, output)
        tokens = len(output) // 4
        cost = tokens * (15 / 1_000_000)
        return TaskResult(
            task_id=task.id, category=task.category.value, prompt=task.prompt,
            backend="premium", output=output, score=score, score_reason=reason,
            escalated=False, local_tokens=0, premium_tokens=tokens,
            premium_cost_usd=cost, latency_sec=round(latency, 2),
        )
    except Exception as e:
        return TaskResult(
            task_id=task.id, category=task.category.value, prompt=task.prompt,
            backend="premium", output="", score=1, score_reason="Error",
            escalated=False, local_tokens=0, premium_tokens=0,
            premium_cost_usd=0.0, latency_sec=0.0, error=str(e),
        )


def run_benchmark(
    categories: list[str] | None = None,
    include_premium: bool = False,
    premium_model: str = "claude-sonnet-4-5",
    save: bool = True,
) -> BenchReport:
    """Run all (or filtered) benchmark tasks, print a Rich progress bar, and return a BenchReport."""
    tasks = TASKS
    if categories:
        tasks = [t for t in tasks if t.category.value in categories]

    report = BenchReport()
    _RESULTS_DIR.mkdir(exist_ok=True)

    console.print(f"\n[bold]Running benchmark:[/bold] {len(tasks)} tasks"
                  f"  |  premium comparison: {'yes' if include_premium else 'no'}\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        for bench_task in tasks:
            pid = progress.add_task(
                f"[cyan]{bench_task.id}[/cyan] local  — {bench_task.prompt[:55]}…",
                total=None,
            )
            result = _run_local(bench_task)
            report.results.append(result)
            progress.remove_task(pid)

            if include_premium:
                pid2 = progress.add_task(
                    f"[magenta]{bench_task.id}[/magenta] premium — {bench_task.prompt[:50]}…",
                    total=None,
                )
                p_result = _run_premium(bench_task, model=premium_model)
                report.results.append(p_result)
                progress.remove_task(pid2)

    if save:
        out_path = _RESULTS_DIR / "latest.json"
        with out_path.open("w") as f:
            json.dump([asdict(r) for r in report.results], f, indent=2)
        console.print(f"[dim]Results saved to {out_path}[/dim]")

    return report
