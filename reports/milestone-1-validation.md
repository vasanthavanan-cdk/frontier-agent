# Milestone 1 — Gemma 4 12B Validation Report

**Date**: 2026-06-07  
**Model**: `gemma4:12b` (7.6GB, Q4_K_M via llama.cpp)  
**Hardware**: Mac Mini M4 Pro, 24GB unified memory  
**Ollama**: 0.30.6 (official app, not Homebrew — Homebrew ships MLX-only which requires 32GB)

---

## Setup Notes

- Homebrew `ollama` package ships **MLX backend only** — requires 32GB minimum, unusable on 24GB.
- Fix: install official Ollama.app from ollama.com, which includes `llama-server` (GGUF/llama.cpp) + MLX backends.
- Model stored at `~/.ollama/models/` and persists across Ollama reinstalls.
- Server start command: `OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve`

---

## Validation Results

| # | Test | Criteria | Result | Score |
|---|------|----------|--------|-------|
| 1 | Tool Calling | Emits valid JSON `{"tool": ..., "arguments": {...}}` | Correct schema, correct query | PASS (5/5) |
| 2 | Code Generation | Runnable async Python with exponential backoff | Correct structure, idiomatic | PASS (4/5)* |
| 3 | Planning / Decomposition | Numbered steps with explicit dependency references | 12-step plan, multi-dependency steps correct | PASS (5/5) |
| 4 | Multimodal Vision | Accurate description of image content | Correctly described colors, patterns, textures | PASS (5/5) |
| 5 | Instruction Following | All 4 constraints met simultaneously | All constraints followed precisely | PASS (5/5) |

**Overall: 5/5 PASS**

*Test 2 minor note: uses `httpx.get` (sync) inside async function. Structurally correct; production would use `httpx.AsyncClient`. Not a blocker.

---

## Decision

**Gemma 4 12B confirmed as Frontier Agent orchestrator.**

Proceed to Milestone 2: build LangGraph orchestration core.

---

## Observed Performance

- First token latency: ~3-5 seconds (model load)
- Sustained generation: ~25-35 tok/s on M4 Pro with GGUF Q4_K_M
- Thinking mode activates automatically on reasoning tasks (visible in output)
- REST API (`http://localhost:11434/api/generate`) works correctly for multimodal inputs

---

## Additional Models

No additional models needed at this stage. Gemma 4 12B handles all tested task types.

Pull `qwen2.5-coder:14b` only if complex coding tasks show insufficient quality in Milestone 2 testing.
