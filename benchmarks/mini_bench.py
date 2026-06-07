"""Run a 5-task representative benchmark and print results."""
from __future__ import annotations

import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.tasks.suite import TASKS, Category
from benchmarks.judge import judge
from benchmarks.runner import TaskResult, BenchReport, _RESULTS_DIR

# Pick 2 coding, 2 reasoning, 1 research
MINI_IDS = {"code-01", "code-03", "reason-01", "reason-04", "research-03"}
MINI_TASKS = [t for t in TASKS if t.id in MINI_IDS]


def run_local(task):
    from frontier_agent.orchestrator.graph import run as fa_run
    start = time.perf_counter()
    try:
        state = fa_run(task=task.prompt, workflow_name=task.workflow)
        latency = time.perf_counter() - start
        score, reason = judge(task, state.final_output)
        return TaskResult(
            task_id=task.id, category=task.category.value, prompt=task.prompt,
            backend="local", output=state.final_output, score=score,
            score_reason=reason, escalated=state.escalation_approved,
            local_tokens=state.token_usage.local_tokens,
            premium_tokens=state.token_usage.premium_tokens,
            premium_cost_usd=state.token_usage.premium_cost_usd,
            latency_sec=round(latency, 2),
        )
    except Exception as e:
        return TaskResult(
            task_id=task.id, category=task.category.value, prompt=task.prompt,
            backend="local", output="", score=1, score_reason=f"Error: {e}",
            escalated=False, local_tokens=0, premium_tokens=0,
            premium_cost_usd=0.0, latency_sec=0.0, error=str(e),
        )


if __name__ == "__main__":
    report = BenchReport()
    _RESULTS_DIR.mkdir(exist_ok=True)

    for i, task in enumerate(MINI_TASKS, 1):
        print(f"\n[{i}/{len(MINI_TASKS)}] {task.id} ({task.category.value})")
        result = run_local(task)
        report.results.append(result)
        print(f"  score={result.score}/5  escalated={result.escalated}  "
              f"latency={result.latency_sec:.1f}s  tokens={result.local_tokens}")
        print(f"  judge: {result.score_reason}")

    out = _RESULTS_DIR / "mini.json"
    with out.open("w") as f:
        json.dump([r.__dict__ for r in report.results], f, indent=2)

    print("\n" + "="*60)
    report.print_summary()
