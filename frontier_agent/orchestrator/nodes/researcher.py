"""Research node: gathers current web context before the main pipeline starts.

When `state.research_enabled` is True, this node first classifies task intent:

  * reasoning/coding tasks      → skipped entirely (local model is competent
                                  on closed-context work; no retrieval needed)
  * current-facts tasks         → run the retrieval lane:
      1. generate targeted search queries from the task
      2. search via the current-facts lane (Tavily if TAVILY_API_KEY is set —
         clean, recency-ranked, pre-extracted — else DuckDuckGo with
         stale-URL filtering)
      3. use pre-extracted content, or fetch + extract HTML as a fallback
      4. synthesize findings into a Research Brief

The brief is stored in `state.research_context` and prepended to every
downstream node's user message, grounding responses in current sources.

If research is disabled, not needed, or fails at any step, the node passes
state through unchanged — downstream nodes degrade gracefully.
"""
from __future__ import annotations

from rich.console import Console

from ..state import AgentState
from ...research.classifier import classify_intent
from ...research.query_generator import generate_queries
from ...research.searcher import search, search_current_facts
from ...research.fetcher import fetch_all
from ...research.synthesizer import synthesize
from ...logger import get_logger

console = Console()
log = get_logger(__name__)


def researcher_node(state: AgentState) -> AgentState:
    """Web research node: search → fetch → synthesize → inject context into state."""
    if not state.research_enabled:
        return state

    # Tool-calling path: the model drives its own retrieval in tool_agent_node.
    # Skip pre-fetch entirely so the model sees a clean slate and calls tools itself.
    if state.tool_calling_enabled:
        log.info("[%s] RESEARCH skipped — tool_calling_enabled (tool_agent handles retrieval)", state.task_id)
        return state

    # Two-lane routing: only "current facts" questions (latest version, news,
    # prices) need live web retrieval. Pure reasoning/coding tasks are answered
    # from the local model's own competence — no retrieval, no latency.
    intent = classify_intent(state.original_input)
    if intent != "current_facts":
        log.info("[%s] RESEARCH skipped — intent=%s (reasoning lane)", state.task_id, intent)
        return state

    console.print("[dim]→ Researcher gathering current sources...[/dim]")
    log.info("[%s] RESEARCH START  intent=current_facts  task=%s", state.task_id, state.original_input[:80])

    try:
        model = state.model_assignments.get("planner", "qwen2.5-coder:7b")

        # 1 — generate targeted queries
        queries = generate_queries(state.original_input, model=model)
        if not queries:
            log.warning("[%s] RESEARCH query generation returned empty list — continuing without context", state.task_id)
            console.print("[dim]  No queries generated — proceeding without web context.[/dim]")
            return state

        log.info("[%s] RESEARCH queries=%s", state.task_id, queries)
        console.print(f"[dim]  Queries: {', '.join(queries[:2])}{'...' if len(queries) > 2 else ''}[/dim]")

        # 2 — search via the current-facts lane (Tavily if configured, else DDG)
        per_query = max(2, state.research_max_sources // len(queries))
        search_results = search_current_facts(queries, max_results_per_query=per_query)
        urls = [r["url"] for r in search_results]
        log.info("[%s] RESEARCH found %d candidate URLs", state.task_id, len(urls))

        # 3 — prefer pre-extracted content (Tavily); otherwise fetch + extract HTML
        fetched = [
            {"url": r["url"], "content": r["content"]}
            for r in search_results
            if r.get("content")
        ][:state.research_max_sources]
        if fetched:
            log.info("[%s] RESEARCH using %d pre-extracted sources (no HTML fetch)", state.task_id, len(fetched))
        else:
            fetched = fetch_all(urls, max_fetch=state.research_max_sources)
            log.info("[%s] RESEARCH fetched %d pages successfully", state.task_id, len(fetched))

        if not fetched:
            # use search snippets as lightweight fallback
            snippet_sources = [
                {"url": r["url"], "content": f"{r['title']}\n{r['snippet']}"}
                for r in search_results
                if r.get("snippet")
            ][:state.research_max_sources]
            if snippet_sources:
                log.info("[%s] RESEARCH using %d search snippets as fallback", state.task_id, len(snippet_sources))
                fetched = snippet_sources
            else:
                log.warning("[%s] RESEARCH no content fetched — continuing without context", state.task_id)
                console.print("[dim]  No pages fetched — proceeding without web context.[/dim]")
                return state

        # 4 — synthesize into a research brief
        brief = synthesize(state.original_input, fetched, model=model)
        sources = [f["url"] for f in fetched]

        log.info(
            "[%s] RESEARCH DONE  brief_chars=%d  sources=%d",
            state.task_id, len(brief), len(sources),
        )
        console.print(f"[green]✓[/green] Research complete — {len(sources)} sources, {len(brief)} chars")

        return AgentState(**{
            **state.model_dump(),
            "research_context": brief,
            "research_sources": sources,
            "research_queries": queries,
        })

    except Exception as e:
        log.warning("[%s] RESEARCH failed: %s — continuing without context", state.task_id, e)
        console.print(f"[yellow]⚠ Research failed ({e}) — proceeding with local knowledge.[/yellow]")
        return state
