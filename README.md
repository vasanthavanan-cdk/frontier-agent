# Frontier Agent

A local-first AI orchestration platform. Runs tasks through **Gemma 4 12B on your Mac** and escalates to Claude or other premium models **only with your explicit approval**.

Built to reduce Claude token consumption and GitHub Copilot AI credit usage without sacrificing quality.

---

## How It Works

```
Your Task
   │
   ▼
Planner (Gemma 4 12B)          ← classifies task, decomposes into steps
   │
   ▼
Coder / Researcher / Reviewer  ← local specialist agents
   │
   ▼
Confidence Scorer              ← behavioural signals, not model self-report
   │
   ├── score ≥ threshold ──────► Return output  ($0.00)
   │
   └── score < threshold ──────► Escalation Gate (YOU decide)
                                        │
                                 Approve? [y/N]
                                        │
                                   Yes  ▼
                                  Claude / GPT
```

No premium model is called without a blocking prompt showing you the confidence breakdown, reason, and estimated cost.

---

## Benchmark Results

Tested on Mac Mini M4 Pro 24 GB — all tasks ran fully locally:

| Category | Tasks | Avg Score | Local ≥4 Rate | Escalations | Cost |
|----------|-------|-----------|---------------|-------------|------|
| Coding | 2 | 5.0/5 | 100% | 0 | $0.00 |
| Reasoning | 2 | 4.0/5 | 50% | 0 | $0.00 |
| Research | 1 | 5.0/5 | 100% | 0 | $0.00 |
| **Overall** | **5** | **4.6/5** | **80%** | **0** | **$0.00** |

Equivalent direct-Claude cost for the same 5 tasks: ~$0.057

---

## Requirements

| Requirement | Notes |
|-------------|-------|
| Mac (Apple Silicon) or Linux | Tested on M4 Pro; Linux amd64/arm64 supported |
| 16 GB unified memory (24 GB recommended) | Gemma 4 12B needs ~8 GB headroom |
| Python 3.11+ | 3.14 recommended |
| `ANTHROPIC_API_KEY` | Only needed if you approve escalation to Claude |

Ollama is installed automatically by `frontier start` — no manual download needed.

---

## New Machine Setup

```bash
# 1. Clone and install
git clone https://github.com/vasanthavanan-cdk/frontier-agent.git
cd frontier-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Make frontier available everywhere (no venv activation needed after this)
frontier install-global --dir ~/.local/bin
source ~/.zshrc          # or ~/.bash_profile on bash

# 3. Configure environment (only needed for Claude escalation)
cp .env.example .env
# edit .env and add: ANTHROPIC_API_KEY=sk-ant-...

# 4. Start — installs Ollama and pulls gemma4:12b automatically if missing
frontier start
```

That's it. `frontier start` handles the rest:

| What it checks | What it does if missing |
|----------------|------------------------|
| `ollama` binary | Downloads from GitHub releases to `~/.local/bin` |
| Ollama server | Starts it with performance flags in the background |
| `gemma4:12b` model | Pulls it (~7.6 GB, one-time download) |

> **Note for macOS users who installed Ollama via Homebrew:** `brew install ollama` ships the MLX-only backend which requires 32 GB minimum and will fail on 16–24 GB machines. Run `brew uninstall ollama` first, then let `frontier start` install the correct binary.

---

## CLI Usage

```bash
# Run any task (auto-detects workflow)
frontier run "Refactor this auth module to use JWT"

# Choose a workflow explicitly
frontier run --workflow coding   "Add unit tests for UserService"
frontier run --workflow research "Explain the CAP theorem"
frontier run --workflow review   "Review this middleware for security issues"

# System status
frontier status

# List available workflows
frontier workflows

# Model management
frontier models status          # show registry — downloaded vs available
frontier models pull coding     # pull qwen2.5-coder:14b if coding quality is lacking
frontier models pull reasoning  # pull deepseek-r1:14b for complex reasoning

# Benchmarking
frontier bench                  # run 5-task mini-bench locally
frontier bench --premium        # compare against Claude (requires ANTHROPIC_API_KEY)
frontier bench --category coding
```

---

## Connect to Your AI Agent

Frontier Agent implements the **Model Context Protocol (MCP)** over stdio. Any MCP-compatible agent can use it as a tool — route tasks through the local pipeline before spending cloud tokens.

### GitHub Copilot (VS Code)

Requires VS Code 1.99+ with the GitHub Copilot extension.

The `.vscode/mcp.json` file is already included in this repo:

```json
{
  "servers": {
    "frontier-agent": {
      "type": "stdio",
      "command": "${workspaceFolder}/.venv/bin/python",
      "args": ["-m", "frontier_agent.mcp_server"]
    }
  }
}
```

Open VS Code in this project folder and Copilot will detect the server automatically. In Copilot Chat, switch to **Agent mode** (`@` menu) and `frontier-agent` will appear as a tool.

To enable it globally for all projects, add to your VS Code `settings.json`:

```json
"mcp": {
  "servers": {
    "frontier-agent": {
      "type": "stdio",
      "command": "/absolute/path/to/frontier-agent/.venv/bin/python",
      "args": ["-m", "frontier_agent.mcp_server"]
    }
  }
}
```

### Claude Code CLI

```bash
claude mcp add frontier-agent -- \
  /absolute/path/to/frontier-agent/.venv/bin/python \
  -m frontier_agent.mcp_server
```

Frontier Agent tools will appear automatically in any Claude Code session.

### Cursor

Add to `.cursor/mcp.json` in your project (or `~/.cursor/mcp.json` for global):

```json
{
  "mcpServers": {
    "frontier-agent": {
      "command": "/absolute/path/to/frontier-agent/.venv/bin/python",
      "args": ["-m", "frontier_agent.mcp_server"]
    }
  }
}
```

### Windsurf

Add to `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "frontier-agent": {
      "command": "/absolute/path/to/frontier-agent/.venv/bin/python",
      "args": ["-m", "frontier_agent.mcp_server"]
    }
  }
}
```

### Continue.dev

Add to `~/.continue/config.json` under `mcpServers`:

```json
{
  "mcpServers": [
    {
      "name": "frontier-agent",
      "command": "/absolute/path/to/frontier-agent/.venv/bin/python",
      "args": ["-m", "frontier_agent.mcp_server"]
    }
  ]
}
```

### Any other MCP client

The server speaks standard **MCP stdio** transport. Use:

- **Command**: `/absolute/path/to/frontier-agent/.venv/bin/python`
- **Args**: `["-m", "frontier_agent.mcp_server"]`
- **Transport**: `stdio`

---

## MCP Tools Reference

| Tool | Description |
|------|-------------|
| `run_task` | Run a task through the local pipeline. Returns output, confidence scores, escalation status, token usage. |
| `list_workflows` | List available workflows with model assignments and thresholds. |
| `models_status` | Show which local models are downloaded with disk usage and roles. |
| `get_result` | Retrieve a previous result by `task_id`. |

### `run_task` parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | string | required | The task or question to process |
| `workflow` | string | `"coding"` | Workflow to use: `coding`, `research`, or `review` |

---

## Escalation Gate

When local confidence falls below threshold after retries, you get a blocking prompt:

```
╭──────────────── ⚠  ESCALATION REQUESTED ─────────────────╮
│                                                           │
│  Step       Score                Threshold   Status       │
│  planner    ████████████████░░░░ 0.82  0.70    OK         │
│  coder      ████████░░░░░░░░░░░░ 0.41  0.75   LOW         │
│  reviewer   ███████░░░░░░░░░░░░░ 0.38  0.80   LOW         │
│                                                           │
╰───────────────────────────────────────────────────────────╯
╭──────────────────── Cost Estimate ────────────────────────╮
│  Local model        gemma4:12b                            │
│  Escalation target  claude-sonnet-4-5                     │
│  Reason             Confidence below threshold after      │
│                     2 retry attempts                      │
│  Estimated tokens   ~3,200                                │
│  Estimated cost     ~$0.0057 USD                          │
╰───────────────────────────────────────────────────────────╯

Approve escalation to premium model? [y/N]:
```

This prompt **cannot be bypassed** — `require_human_approval: true` is enforced at schema load time and raises a `ConfigurationError` if set to `false`.

---

## Customising Workflows

Workflows are plain YAML. Edit them directly — changes take effect on the next run with no restart.

```yaml
# frontier_agent/workflows/definitions/default_coding.yaml
models:
  planner:  "gemma4:12b"
  coder:    "qwen2.5-coder:14b"   # swap to specialist after pulling
  reviewer: "gemma4:12b"
  fallback_premium: "claude-sonnet-4-5"

confidence_thresholds:
  coder:    0.75   # lower = fewer escalations
  reviewer: 0.78

escalation:
  require_human_approval: true   # cannot be set to false
  max_local_retries: 2
```

---

## Model Stack

Start with just `gemma4:12b`. Pull specialists only when you observe a real quality gap.

| Model | Size | Role | Pull when |
|-------|------|------|-----------|
| `gemma4:12b` | 7.6 GB | Default orchestrator + all roles | **Now (required)** |
| `qwen2.5-coder:14b` | ~9 GB | Coding specialist | Coding quality insufficient |
| `deepseek-r1:14b` | ~9 GB | Reasoning specialist | Complex multi-step reasoning fails |
| `qwen3:14b` | ~9 GB | Fallback orchestrator | Gemma 4 tool-calling unreliable |

Maximum realistic footprint: **~27 GB** (vs ~140 GB if all were pre-downloaded).

---

## Project Structure

```
frontier-agent/
├── frontier_agent/
│   ├── cli.py                    # Typer CLI — frontier run/status/models/bench
│   ├── mcp_server.py             # MCP server — exposes 4 tools over stdio
│   ├── orchestrator/
│   │   ├── graph.py              # LangGraph multi-agent graph
│   │   ├── state.py              # AgentState (Pydantic v2)
│   │   ├── confidence.py         # Composite confidence scorer
│   │   ├── escalation.py         # Human approval gate (Rich UI)
│   │   ├── router.py             # Conditional edge logic
│   │   └── nodes/                # planner, coder, reviewer, premium
│   ├── workflows/
│   │   ├── schema.py             # Pydantic WorkflowSpec
│   │   ├── loader.py             # YAML → LangGraph compiler
│   │   └── definitions/          # default_coding/research/review.yaml
│   └── models/
│       └── registry.py           # Known models with roles and sizes
├── benchmarks/
│   ├── runner.py                 # Benchmark execution engine
│   ├── judge.py                  # Automated quality scoring (Gemma 4 judge)
│   └── tasks/suite.py            # 20 standardised tasks
├── reports/
│   └── milestone-1-validation.md # Gemma 4 12B validation results
├── .vscode/mcp.json              # GitHub Copilot MCP config (auto-detected)
└── pyproject.toml
```

---

## Troubleshooting

**`llama-server binary not found` when starting Ollama**
You installed Ollama via Homebrew (`brew install ollama`), which ships the MLX-only backend and requires 32 GB minimum. Run `brew uninstall ollama`, then `frontier start` will download the correct binary automatically.

**Escalation gate appears for simple tasks**
Lower the confidence thresholds in the relevant workflow YAML (e.g. set `coder: 0.60`). No restart needed.

**MCP server not appearing in VS Code Copilot**
Make sure you opened VS Code with the `frontier-agent` folder as the workspace root (not a parent folder). The `.vscode/mcp.json` is project-scoped.

**`frontier` command not found after install**
Activate the venv: `source .venv/bin/activate`. Or reinstall: `pip install -e .`.

**Ollama connection refused**
Start the server: `OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve`

---

## License

Apache 2.0
