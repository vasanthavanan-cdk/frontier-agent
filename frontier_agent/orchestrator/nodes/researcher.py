"""Research node: gathers current web context before the main pipeline starts.

When `state.research_enabled` is True, this node:
  1. Uses Gemma to generate targeted search queries from the task
  2. Searches DuckDuckGo for relevant URLs, ranked by domain authority
  3. Fetches and extracts content from the top pages
  4. Synthesizes findings into a Research Brief via Gemma

The brief is stored in `state.research_context` and prepended to every
downstream node's user message, grounding responses in current sources.

If research is disabled or fails at any step, the node passes state
through unchanged — downstream nodes degrade gracefully to local knowledge.
"""
from __future__ import annotations

from rich.console import Console

from ..state import AgentState
from ...research.query_generator import generate_queries
from ...research.searcher import search
from ...research.fetcher import fetch_all
from ...research.synthesizer import synthesize
from ...logger import get_logger

console = Console()
log = get_logger(__name__)


def researcher_node(state: AgentState) -> AgentState:
    """Web research node: search → fetch → synthesize → inject context into state."""
    if not state.research_enabled:
        return state

    console.print("[dim]→ Researcher gathering current sources...[/dim]")
    log.info("[%s] RESEARCH START  task=%s", state.task_id, state.original_input[:80])

    try:
        model = state.model_assignments.get("planner", "gemma4:12b")

        # 1 — generate targeted queries
        queries = generate_queries(state.original_input, model=model)
        log.info("[%s] RESEARCH queries=%s", state.task_id, queries)
        console.print(f"[dim]  Queries: {', '.join(queries[:2])}{'...' if len(queries) > 2 else ''}[/dim]")

        # 2 — search the web
        search_results = search(queries, max_results_per_query=max(2, state.research_max_sources // len(queries)))
        urls = [r["url"] for r in search_results]
        log.info("[%s] RESEARCH found %d candidate URLs", state.task_id, len(urls))

        # 3 — fetch and extract content
        fetched = fetch_all(urls, max_fetch=state.research_max_sources)
        log.info("[%s] RESEARCH fetched %d pages successfully", state.task_id, len(fetched))

        if not fetched:
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
