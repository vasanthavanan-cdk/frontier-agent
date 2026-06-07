# Frontier Agent

A local-first AI orchestration platform that routes tasks to offline models on your Mac, escalating to Claude or other premium models **only with your explicit approval**.

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

No premium model is ever called without a blocking prompt showing you the confidence breakdown, reason, and estimated cost.

---

## Benchmark Results

Tested on Mac Mini M4 Pro 24GB — all tasks ran fully locally:

| Category | Tasks | Avg Score | Local ≥4 Rate | Escalations | Cost |
|----------|-------|-----------|---------------|-------------|------|
| Coding | 2 | 5.0/5 | 100% | 0 | $0.00 |
| Reasoning | 2 | 4.0/5 | 50% | 0 | $0.00 |
| Research | 1 | 5.0/5 | 100% | 0 | $0.00 |
| **Overall** | **5** | **4.6/5** | **80%** | **0** | **$0.00** |

Equivalent direct-Claude cost for the same tasks: ~$0.057

---

## Requirements

- Mac with Apple Silicon (M1/M2/M3/M4)
- 16GB+ unified memory (24GB recommended for 32B models)
- [Ollama](https://ollama.com) — official app installer (not Homebrew — see note below)
- Python 3.11+

> **Note on Ollama installation:** The Homebrew `ollama` package ships the MLX backend only, which requires 32GB minimum. For 16–24GB systems, install from [ollama.com](https://ollama.com/download/mac) to get the full llama.cpp GGUF backend.

---

## Setup

```bash
# 1. Clone
git clone https://github.com/vasanthavanan-cdk/frontier-agent.git
cd frontier-agent

# 2. Install Ollama from https://ollama.com/download/mac
# Then start the server
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve &

# 3. Pull the default model (~7.6 GB)
ollama pull gemma4:12b

# 4. Create Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 5. Configure environment
cp .env.example .env
# Add ANTHROPIC_API_KEY only if you want Claude escalation enabled
```

---

## Usage

```bash
# Run any task through the local pipeline
frontier run "Write a Python function that reverses a linked list"

# Choose a specific workflow
frontier run --workflow research "Explain the CAP theorem"
frontier run --workflow review "Review this auth middleware for security issues"

# System status
frontier status

# See available workflows
frontier workflows

# Model management
frontier models status          # show registry — downloaded vs available
frontier models pull coding     # pull qwen2.5-coder:14b if coding quality is lacking
frontier models pull reasoning  # pull deepseek-r1:14b for complex reasoning tasks

# Run benchmark suite
frontier bench                  # local only
frontier bench --premium        # compare against Claude (requires ANTHROPIC_API_KEY)
frontier bench --category coding
```

---

## Escalation Gate

When local confidence falls below the configured threshold after retries, you get a blocking prompt:

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

Workflows are plain YAML files — edit them directly, changes take effect on the next run with no restart needed.

```yaml
# frontier_agent/workflows/definitions/default_coding.yaml
models:
  planner:  "gemma4:12b"
  coder:    "qwen2.5-coder:14b"   # swap to specialist after pulling
  reviewer: "gemma4:12b"
  fallback_premium: "claude-sonnet-4-5"

confidence_thresholds:
  coder: 0.75      # lower = fewer escalations, higher = stricter quality gate
  reviewer: 0.78

escalation:
  require_human_approval: true   # cannot be set to false
  max_local_retries: 2
```

---

## Model Stack

| Model | Size | Role | Pull when |
|-------|------|------|-----------|
| `gemma4:12b` | 7.6 GB | Default orchestrator + all roles | Now (required) |
| `qwen2.5-coder:14b` | ~9 GB | Coding specialist | Coding quality insufficient |
| `deepseek-r1:14b` | ~9 GB | Reasoning specialist | Complex multi-step reasoning fails |
| `qwen3:14b` | ~9 GB | Fallback orchestrator | Gemma 4 tool-calling unreliable |

Maximum realistic footprint: **~27 GB** (vs ~140 GB if all were pre-downloaded).

---

## Project Structure

```
frontier-agent/
├── frontier_agent/
│   ├── cli.py                          # Typer CLI — frontier run/status/models/bench
│   ├── orchestrator/
│   │   ├── graph.py                    # LangGraph multi-agent graph
│   │   ├── state.py                    # AgentState (Pydantic v2)
│   │   ├── confidence.py               # Composite confidence scorer
│   │   ├── escalation.py               # Human approval gate (Rich UI)
│   │   ├── router.py                   # Conditional edge logic
│   │   └── nodes/                      # planner, coder, reviewer, premium
│   ├── workflows/
│   │   ├── schema.py                   # Pydantic WorkflowSpec
│   │   ├── loader.py                   # YAML → LangGraph compiler
│   │   └── definitions/                # default_coding/research/review.yaml
│   └── models/
│       └── registry.py                 # Known models with roles and sizes
├── benchmarks/
│   ├── runner.py                       # Benchmark execution engine
│   ├── judge.py                        # Automated quality scoring (local LLM judge)
│   └── tasks/suite.py                  # 20 standardised tasks (coding/reasoning/research)
├── reports/
│   └── milestone-1-validation.md       # Gemma 4 12B validation results
└── pyproject.toml
```

---

## License

Apache 2.0
