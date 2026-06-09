# Frontier Agent

A local-first AI orchestration platform. Runs tasks through **Qwen2.5-Coder 7B on your Mac** and escalates to Claude or other premium models **only with your explicit approval**.

Built to reduce Claude token consumption and GitHub Copilot AI credit usage without sacrificing quality.

---

## How It Works

The pipeline has two paths depending on the workflow:

### Coding / Review workflow

```
Your Task
   │
   ▼
Planner (Qwen2.5-Coder 7B)     ← decomposes task into steps
   │
   ▼
Coder (Qwen2.5-Coder 7B)       ← implements the plan
   │
   ▼
Confidence Scorer               ← behavioural signals, not model self-report
   │
   ├── score ≥ threshold ──────► Reviewer → Return output  ($0.00)
   │
   └── score < threshold ──────► Escalation Gate (YOU decide)
                                        │
                                 Approve? [y/N]
                                        │
                                   Yes  ▼
                                  Claude / GPT
```

### Research workflow (tool calling)

```
Your Task
   │
   ▼
Tool Agent (Qwen2.5-Coder 7B)  ← model calls web_search / web_fetch tools
   │   ↑                           itself (up to 4 rounds) to find live data
   └───┘  tool results fed back
   │
   ▼
Reviewer → Return output  ($0.00)
```

The research workflow uses **Ollama tool calling** — the model decides what to search and fetch instead of relying on a pre-classifier. Results flow back as tool messages, grounding the answer in live web content.

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
| 16 GB unified memory (24 GB recommended) | Qwen2.5-Coder 7B needs ~5 GB headroom |
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

# 4. Start — installs Ollama and pulls qwen2.5-coder:7b automatically if missing
frontier start
```

That's it. `frontier start` handles the rest:

| What it checks | What it does if missing |
|----------------|------------------------|
| `ollama` binary | Downloads from GitHub releases to `~/.local/bin` |
| Ollama server | Starts it with performance flags in the background |
| `qwen2.5-coder:7b` model | Pulls it (~4.7 GB, one-time download) |

> **Note for macOS users who installed Ollama via Homebrew:** `brew install ollama` ships the MLX-only backend which requires 32 GB minimum and will fail on 16–24 GB machines. Run `brew uninstall ollama` first, then let `frontier start` install the correct binary.

---

## CLI Usage

```bash
# Run any task — choose workflow explicitly
frontier run --workflow coding   "Add unit tests for UserService"
frontier run --workflow research "What is the latest stable version of Node.js?"
frontier run --workflow review   "Review this middleware for security issues"

# System status
frontier status

# List available workflows
frontier workflows

# Model management
frontier models status          # show registry — downloaded vs available
frontier models pull coding     # pull qwen2.5-coder:14b if coding quality is lacking
frontier models pull reasoning  # pull deepseek-r1:14b for complex reasoning

# View pipeline logs
frontier logs                   # last 50 lines
frontier logs --follow          # stream live (like tail -f)

# Benchmarking
frontier bench                  # run 5-task mini-bench locally
frontier bench --premium        # compare against Claude (requires ANTHROPIC_API_KEY)
frontier bench --category coding
```

---

## Web Research & Tool Calling

The `research` workflow gives the model live web access via **Ollama tool calling**. qwen2.5-coder:7b calls `web_search` and `web_fetch` tools during generation — no fixed pre-classifier, no pre-synthesized brief.

```
frontier run --workflow research "what is the latest stable version of Python?"
# Watch the console:
#   → Tool Agent searching the web...
#   ↳ web_search(query='latest stable Python version 2025')
#   ✓ Tool Agent done — 1 tool round(s), confidence=0.74
```

### Search backends (priority order)

| Backend | Setup | Cost | Notes |
|---------|-------|------|-------|
| **SearXNG** | Docker (15 min) | Free | Recommended — self-hosted, aggregates 70+ engines, no rate limits |
| **Tavily** | API key | Free tier | 1,000 searches/month free |
| **DuckDuckGo** | None | Free | Default fallback, scraping-based |

**To enable SearXNG** (recommended for production use):

```bash
# Start SearXNG locally
docker run -d -p 8080:8080 searxng/searxng

# Add to .env
SEARXNG_BASE_URL=http://localhost:8080
```

The agent routes automatically — SearXNG when running, Tavily if API key set, DuckDuckGo as fallback.

### View tool call trace

```bash
frontier logs --follow
# or filter to tool calls only:
tail -f ~/.frontier/mcp.log | grep -E "TOOL_CALL|TOOL_AGENT"
```

---

## Connect to Your AI Agent

Frontier Agent implements the **Model Context Protocol (MCP)** over stdio. Any MCP-compatible agent can use it as a tool — route tasks through the local pipeline before spending cloud tokens.

### Claude Code CLI

```bash
claude mcp add frontier-agent -- \
  /absolute/path/to/frontier-agent/.venv/bin/python \
  -m frontier_agent.mcp_server
```

Frontier Agent tools will appear automatically in any Claude Code session.

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
│  Local model        qwen2.5-coder:7b                      │
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
  planner:  "qwen2.5-coder:7b"
  coder:    "qwen2.5-coder:14b"   # swap to specialist after pulling
  reviewer: "qwen2.5-coder:7b"
  fallback_premium: "claude-sonnet-4-5"

confidence_thresholds:
  coder:    0.75   # lower = fewer escalations
  reviewer: 0.78

escalation:
  require_human_approval: true   # cannot be set to false
  max_local_retries: 2
```

```yaml
# frontier_agent/workflows/definitions/default_research.yaml
research:
  enabled: true
  max_sources: 6
  tool_calling: true    # model calls web_search/web_fetch tools itself
```

---

## Model Stack

Start with just `qwen2.5-coder:7b`. Pull specialists only when you observe a real quality gap.

| Model | Size | Role | Pull when |
|-------|------|------|-----------|
| `qwen2.5-coder:7b` | 4.7 GB | Default orchestrator + all roles | **Now (required)** |
| `qwen2.5-coder:14b` | ~9 GB | Coding specialist | Coding quality insufficient |
| `deepseek-r1:14b` | ~9 GB | Reasoning specialist | Complex multi-step reasoning fails |
| `qwen3:14b` | ~9 GB | Fallback orchestrator | Harder planning / tool-calling tasks |

Maximum realistic footprint: **~27 GB** (vs ~140 GB if all were pre-downloaded).

---

## Project Structure

```
frontier-agent/
├── frontier_agent/
│   ├── cli.py                    # Typer CLI — frontier run/status/models/bench/logs
│   ├── mcp_server.py             # MCP server — exposes 4 tools over stdio
│   ├── config.py                 # Settings (OLLAMA_BASE_URL, SEARXNG_BASE_URL, etc.)
│   ├── orchestrator/
│   │   ├── graph.py              # LangGraph multi-agent graph (two-path routing)
│   │   ├── state.py              # AgentState (Pydantic v2)
│   │   ├── confidence.py         # Composite confidence scorer
│   │   ├── escalation.py         # Human approval gate (Rich UI)
│   │   ├── router.py             # Conditional edge logic
│   │   └── nodes/
│   │       ├── base.py           # Shared call_local_model helper
│   │       ├── planner.py
│   │       ├── coder.py
│   │       ├── reviewer.py
│   │       ├── researcher.py     # Pre-fetch pipeline (coding/review path)
│   │       ├── tool_agent.py     # Agentic tool-calling loop (research path)
│   │       └── premium.py        # Claude escalation node
│   ├── tools/
│   │   └── web_tools.py          # web_search + web_fetch LangChain tools
│   ├── research/
│   │   ├── searcher.py           # SearXNG → Tavily → DuckDuckGo routing
│   │   ├── fetcher.py            # httpx + trafilatura page extraction
│   │   ├── query_generator.py    # Ollama-based query generation
│   │   ├── classifier.py         # Intent classifier (reasoning vs current_facts)
│   │   └── synthesizer.py        # Research brief synthesis
│   ├── workflows/
│   │   ├── schema.py             # Pydantic WorkflowSpec
│   │   ├── loader.py             # YAML → AgentState compiler
│   │   └── definitions/          # default_coding/research/review.yaml
│   └── models/
│       └── registry.py           # Known models with roles and sizes
├── benchmarks/
│   ├── runner.py                 # Benchmark execution engine
│   ├── judge.py                  # Automated quality scoring
│   └── mini_bench.py
├── reports/
│   └── milestone-1-validation.md
├── .env.example                  # Environment variables template
├── .vscode/mcp.json              # GitHub Copilot MCP config (auto-detected)
└── pyproject.toml
```

---

## Troubleshooting

**Tool calling returns JSON text instead of executing tools**
qwen2.5-coder:7b outputs tool calls as JSON text rather than using Ollama's structured tool_calls field. This is handled automatically via a content-based parser in `tool_agent.py` — no action needed.

**Research workflow not fetching live data**
Make sure you're using `--workflow research`. The coding workflow intentionally skips web retrieval. Check that Ollama is running: `frontier mcp-status`.

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
