"""
Frontier Agent MCP Server

Exposes the local AI pipeline as MCP tools so any MCP-compatible agent
(Claude Code, Cursor, Windsurf, Continue.dev, Zed, etc.) can call it.

Tools exposed:
  - run_task          : run a task through the local pipeline
  - list_workflows    : list available YAML workflows
  - models_status     : show downloaded models and registry
  - get_result        : retrieve a previous result by task_id
"""
from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from .logger import get_logger, log_path

log = get_logger(__name__)

# ── in-memory result cache (task_id → result dict) ───────────────────────────
_results: dict[str, dict] = {}

server = Server("frontier-agent")


# ── tool definitions ─────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="run_task",
            description=(
                "Run a task through the Frontier Agent local pipeline. "
                "Uses qwen2.5-coder:7b locally — escalates to a premium model only if "
                "confidence is below threshold AND the user approves. "
                "Returns the output, confidence scores, whether it was escalated, "
                "and token usage. Use this before spending your own tokens."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task or question to process.",
                    },
                    "workflow": {
                        "type": "string",
                        "description": "Workflow to use: coding | research | review. Defaults to coding.",
                        "default": "coding",
                    },
                },
                "required": ["task"],
            },
        ),
        Tool(
            name="list_workflows",
            description="List all available Frontier Agent workflows with their model assignments and confidence thresholds.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="models_status",
            description="Show which local models are downloaded and available, with disk usage and assigned roles.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="get_result",
            description="Retrieve a previous Frontier Agent result by task_id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The task_id returned by a previous run_task call.",
                    }
                },
                "required": ["task_id"],
            },
        ),
    ]


# ── tool handlers ─────────────────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    log.info("TOOL CALL  name=%s  args=%s", name, json.dumps(arguments)[:200])
    if name == "run_task":
        return await _handle_run_task(arguments)
    if name == "list_workflows":
        return await _handle_list_workflows()
    if name == "models_status":
        return await _handle_models_status()
    if name == "get_result":
        return await _handle_get_result(arguments)
    log.warning("TOOL CALL  unknown tool: %s", name)
    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _handle_run_task(args: dict) -> list[TextContent]:
    task = args.get("task", "").strip()
    workflow = args.get("workflow", "coding")

    if not task:
        return [TextContent(type="text", text="Error: 'task' is required.")]

    log.info("run_task  workflow=%s  task=%s", workflow, task[:120])

    # run in a thread — LangGraph graph is synchronous
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_pipeline, task, workflow)

    _results[result["task_id"]] = result
    log.info("run_task COMPLETE  task_id=%s  escalated=%s  cost=$%.4f",
             result["task_id"], result["escalated"], result["premium_cost_usd"])

    lines = [
        "## Frontier Agent Result",
        f"**Task ID**: `{result['task_id']}`",
        f"**Workflow**: {result['workflow']}",
        f"**Escalated to premium**: {'Yes ⚠' if result['escalated'] else 'No ✓'}",
        f"**Local tokens**: {result['local_tokens']:,}  |  "
        f"**Premium tokens**: {result['premium_tokens']:,}  |  "
        f"**Cost**: ${result['premium_cost_usd']:.4f} USD",
        "",
        "### Pipeline Trace",
        "",
        _build_trace(result),
        "",
        f"> Full step-by-step log: `tail -f {log_path()}`",
        "",
        "### Output",
        "",
        result["output"],
    ]

    return [TextContent(type="text", text="\n".join(lines))]


def _build_trace(result: dict) -> str:
    """Format a markdown table showing each pipeline step's confidence and routing decision."""
    scores = result.get("confidence_scores", {})
    thresholds = result.get("confidence_thresholds", {})
    model_assignments = result.get("model_assignments", {})
    retry_counts = result.get("retry_counts", {})
    escalated = result.get("escalated", False)
    escalation_reason = result.get("escalation_reason", "")

    rows = ["| Step | Model | Attempt | Confidence | Threshold | Decision |",
            "|------|-------|---------|------------|-----------|----------|"]

    steps = list(scores.keys())
    for i, step in enumerate(steps):
        score = scores[step]
        threshold = thresholds.get(step, thresholds.get("default", 0.70))
        model = model_assignments.get(step, "qwen2.5-coder:7b")
        attempt = retry_counts.get(step, 1)
        passed = score >= threshold

        if i < len(steps) - 1:
            next_step = steps[i + 1]
            decision = f"→ {next_step}" if passed else "→ retry / escalate"
        else:
            if escalated:
                decision = "→ **ESCALATED** ⚠"
            else:
                decision = "→ **done** ✓"

        confidence_fmt = f"`{score:.3f}`" + (" ✓" if passed else " ✗")
        rows.append(f"| {step} | {model} | {attempt} | {confidence_fmt} | {threshold:.2f} | {decision} |")

    if escalated:
        rows.append(f"| _escalation_ | — | — | — | — | Approved by user |")
        rows.append(f"| _premium_ | {result.get('fallback_premium_model', 'claude')} | — | — | — | → done ✓ |")

    if escalation_reason:
        rows.append(f"\n> **Escalation reason**: {escalation_reason}")

    return "\n".join(rows)


async def _handle_list_workflows() -> list[TextContent]:
    defs_dir = Path(__file__).parent / "workflows" / "definitions"
    import yaml

    lines = ["## Available Workflows\n"]
    for path in sorted(defs_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        lines += [
            f"### {data.get('name', path.stem)}",
            f"- **Description**: {data.get('description', '—')}",
            f"- **Steps**: {', '.join(s['id'] for s in data.get('steps', []))}",
            f"- **Coder model**: {data.get('models', {}).get('coder', '—')}",
            f"- **Coder threshold**: {data.get('confidence_thresholds', {}).get('coder', '—')}",
            f"- **Premium fallback**: {data.get('models', {}).get('fallback_premium', '—')}",
            "",
        ]
    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_models_status() -> list[TextContent]:
    from .models.registry import REGISTRY

    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    downloaded = set()
    for line in result.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            downloaded.add(parts[0])

    lines = ["## Local Model Registry\n",
             "| Model | Downloaded | Disk | Roles | Notes |",
             "|-------|-----------|------|-------|-------|"]
    for tag, info in REGISTRY.items():
        dl = "✓" if tag in downloaded else "–"
        lines.append(
            f"| `{tag}` | {dl} | ~{info.size_gb:.0f} GB "
            f"| {', '.join(info.roles)} | {info.description} |"
        )

    if result.returncode != 0:
        lines.append("\n⚠ Ollama server not running — start with: `ollama serve`")

    return [TextContent(type="text", text="\n".join(lines))]


async def _handle_get_result(args: dict) -> list[TextContent]:
    task_id = args.get("task_id", "")
    result = _results.get(task_id)
    if not result:
        return [TextContent(type="text", text=f"No result found for task_id `{task_id}`.")]
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


# ── pipeline runner (sync, called in executor) ────────────────────────────────

def _run_pipeline(task: str, workflow: str) -> dict:
    from .orchestrator.graph import run

    state = run(task=task, workflow_name=workflow)
    return {
        "task_id": state.task_id,
        "workflow": state.workflow_name,
        "output": state.final_output,
        "confidence_scores": state.confidence_scores,
        "confidence_thresholds": state.confidence_thresholds,
        "model_assignments": state.model_assignments,
        "retry_counts": state.retry_counts,
        "escalated": state.escalation_approved,
        "escalation_reason": state.escalation_reason,
        "fallback_premium_model": state.fallback_premium_model,
        "local_tokens": state.token_usage.local_tokens,
        "premium_tokens": state.token_usage.premium_tokens,
        "premium_cost_usd": state.token_usage.premium_cost_usd,
    }


# ── entrypoint ────────────────────────────────────────────────────────────────

async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
