"""Tool-calling agent node: model drives web search/fetch via Ollama tool calling.

Replaces the pre-fetch researcher pipeline for the research workflow.
The model decides what to search and fetch — no fixed classifier, no pre-synthesis.
Results are returned as ToolMessages so the model sees them directly.

Graph slot: researcher (no-op) → tool_agent → reviewer → done
Output is stored under agent_outputs["coder"] so the existing router, reviewer,
and done nodes work without modification.
"""
from __future__ import annotations

import json
import re

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from rich.console import Console

from ..state import AgentState, AgentOutput
from ..confidence import compute_confidence
from ...tools.web_tools import WEB_TOOLS
from ...logger import get_logger

console = Console()
log = get_logger(__name__)

_MAX_TOOL_ROUNDS = 4

_SYSTEM = """You are a research assistant with real-time web access.

You have two tools:
- web_search(query): search the web and get a list of URLs with snippets
- web_fetch(url): fetch and read the full content of a specific page

How to use them:
1. Call web_search with a focused query to find relevant pages
2. Call web_fetch on the most promising URL(s) to read the actual content
3. Base your final answer entirely on what the tools return — do not rely on training data
4. If a search returns no useful results, retry with a different, more specific query
5. Once you have enough information from the tools, write your final answer

Always ground your answer in the retrieved content. If the tools return nothing useful,
say so explicitly rather than guessing."""


def _find_tool(name: str):
    """Return the LangChain tool with the given name, or None."""
    return next((t for t in WEB_TOOLS if t.name == name), None)


def _parse_content_tool_call(content: str) -> dict | None:
    """Parse a JSON tool call from model content text.

    qwen2.5-coder:7b outputs tool calls as plain JSON text rather than using
    Ollama's structured tool_calls field. This detects and parses that pattern.
    Handles bare JSON and code-fenced JSON blocks.
    Expected shape: {"name": "tool_name", "arguments": {...}}
    """
    text = content.strip()
    # strip code fences if present
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*```\s*$', '', text, flags=re.MULTILINE)
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "name" in data and "arguments" in data:
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def tool_agent_node(state: AgentState) -> AgentState:
    """LangGraph node: agentic tool-calling loop for web research tasks."""
    console.print("[dim]→ Tool Agent searching the web...[/dim]")

    role = "coder"
    model_tag = state.model_assignments.get(role, "qwen2.5-coder:7b")
    attempt = state.retry_counts.get(role, 0) + 1

    log.info("[%s] TOOL_AGENT START  model=%s  attempt=%d  task=%s",
             state.task_id, model_tag, attempt, state.original_input[:80])

    llm = ChatOllama(
        model=model_tag,
        base_url="http://localhost:11434",
        temperature=0.1,
        num_predict=4096,
        num_ctx=16384,
        keep_alive="10m",
    )
    llm_with_tools = llm.bind_tools(WEB_TOOLS)

    messages: list = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=state.original_input),
    ]

    last_response: AIMessage | None = None
    tool_rounds = 0

    for round_num in range(_MAX_TOOL_ROUNDS):
        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            log.error("[%s] TOOL_AGENT invoke failed round=%d: %s", state.task_id, round_num, e)
            break

        messages.append(response)
        last_response = response

        # --- Path 1: native Ollama structured tool_calls (preferred) ---
        if response.tool_calls:
            tool_rounds += 1
            log.info("[%s] TOOL_AGENT round=%d  native_tool_calls=%d", state.task_id, round_num, len(response.tool_calls))

            for tc in response.tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {})
                tool_call_id = tc.get("id", f"call_{tool_rounds}")

                log.info("[%s] TOOL_CALL  tool=%s  args=%s", state.task_id, tool_name, tool_args)
                console.print(f"[dim]  ↳ {tool_name}({', '.join(f'{k}={v!r}' for k, v in tool_args.items())})[/dim]")

                tool_fn = _find_tool(tool_name)
                if tool_fn is None:
                    result = f"Unknown tool '{tool_name}'. Available tools: web_search, web_fetch."
                else:
                    try:
                        result = tool_fn.invoke(tool_args)
                    except Exception as e:
                        result = f"Tool error: {e}"
                        log.warning("[%s] TOOL_CALL error  tool=%s: %s", state.task_id, tool_name, e)

                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))
            continue

        # --- Path 2: content-based fallback (qwen2.5-coder:7b outputs JSON text) ---
        parsed = _parse_content_tool_call(response.content)
        if parsed:
            tool_rounds += 1
            tool_name = parsed["name"]
            tool_args = parsed.get("arguments", {})

            log.info("[%s] TOOL_CALL (content-parsed)  tool=%s  args=%s", state.task_id, tool_name, tool_args)
            console.print(f"[dim]  ↳ {tool_name}({', '.join(f'{k}={v!r}' for k, v in tool_args.items())})[/dim]")

            tool_fn = _find_tool(tool_name)
            if tool_fn is None:
                result = f"Unknown tool '{tool_name}'. Available tools: web_search, web_fetch."
            else:
                try:
                    result = tool_fn.invoke(tool_args)
                except Exception as e:
                    result = f"Tool error: {e}"
                    log.warning("[%s] TOOL_CALL error  tool=%s: %s", state.task_id, tool_name, e)

            # inject result as a human message so the model can continue
            messages.append(HumanMessage(
                content=f"Tool result for {tool_name}:\n{result}\n\n"
                        "Now use this information to answer the original question. "
                        "If you need more information, call another tool. Otherwise write your final answer."
            ))
            continue

        # --- No tool call: model is done ---
        log.info("[%s] TOOL_AGENT done after %d rounds (no tool call)", state.task_id, round_num)
        break

    content = (last_response.content or "").strip() if last_response else ""
    if not content:
        content = "The web search did not return useful results for this query."
        log.warning("[%s] TOOL_AGENT no content in final response", state.task_id)

    output = AgentOutput(agent=role, content=content, attempt=attempt)
    confidence = compute_confidence(output, state.task_type, attempt)
    output.confidence = confidence

    state.token_usage.local_tokens += len(content) // 4

    threshold = state.confidence_thresholds.get(role, state.confidence_thresholds.get("default", 0.70))
    verdict = "PASS" if confidence >= threshold else "LOW "
    log.info(
        "[%s] TOOL_AGENT DONE  confidence=%.3f  threshold=%.2f  [%s]  tool_rounds=%d  output_chars=%d",
        state.task_id, confidence, threshold, verdict, tool_rounds, len(content),
    )
    console.print(f"[{'green' if verdict.strip() == 'PASS' else 'yellow'}]✓[/] Tool Agent done — "
                  f"{tool_rounds} tool round(s), confidence={confidence:.2f}")

    return AgentState(**{
        **state.model_dump(),
        "current_step": "reviewer",
        "agent_outputs": {**state.agent_outputs, role: output},
        "confidence_scores": {**state.confidence_scores, role: confidence},
        "retry_counts": {**state.retry_counts, role: state.retry_counts.get(role, 0) + 1},
    })
