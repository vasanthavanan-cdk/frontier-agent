# Frontier Agent — local-first routing

This project exposes a local AI pipeline (qwen2.5-coder:7b on Ollama) via an MCP
server. Use it to handle work locally before spending premium effort.

## When to use the local agent

For **coding** tasks (write/refactor/explain code) and **factual-lookup** tasks
(latest versions, "search the web for X", "who is Y"):

1. First ask me: **"Try the local frontier agent first? (y/n)"**
2. If I say **yes** → call the `frontier-agent` MCP tool **`run_task`**
   (use `workflow: "research"` for factual lookups, `"coding"` for code).
3. Read the result's recommendation:
   - **"✓ Local result is confident"** → relay the output as-is and stop.
   - **"⚠ Escalation recommended"** → the local confidence was low. Treat the
     output as a draft; if it isn't good enough, answer the request yourself.
4. If I say **no** → answer directly without calling the tool.

## When NOT to use it

- Trivial questions, or questions about our current conversation/context.
- Anything where calling a local model would be slower than just answering.

## Notes

- The local agent is **free** and runs on my machine — there is no Anthropic API
  key involved. *You* (via my Claude subscription) are the escalation layer.
- `run_task` runs headless: it never blocks on prompts and never calls a paid
  API. It only ever returns a local draft plus a recommendation.
- Available workflows: `coding`, `research`, `review` (see the `list_workflows`
  tool). Model/health: `models_status`.
