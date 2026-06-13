"""LangChain tool wrappers for web search and page fetch.

These are bound to the local Ollama model in tool_agent_node so qwen2.5-coder:7b
can call them during generation instead of relying on the pre-fetch pipeline.
"""
from __future__ import annotations

from langchain_core.tools import tool

from ..logger import get_logger

log = get_logger(__name__)


@tool
def web_search(query: str, max_results: int = 4) -> str:
    """Search the web for current information. Returns top results with titles, URLs, and snippets.

    Use this to find relevant pages about a topic. Follow up with web_fetch to read a specific page.
    """
    from ..research.searcher import search_current_facts

    log.debug("web_search: query=%s max_results=%d", query, max_results)
    results = search_current_facts([query], max_results_per_query=max_results)
    if not results:
        return "No results found. Try a different query."

    lines: list[str] = []
    for i, r in enumerate(results[:max_results], 1):
        lines.append(
            f"Result {i}: {r['title']}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r.get('snippet', '').strip()}"
        )
    return "\n\n".join(lines)


@tool
def web_fetch(url: str) -> str:
    """Fetch and extract the main text content from a web page URL.

    Use this after web_search to read the full content of a specific page.
    Returns extracted text (up to 4000 characters).
    """
    from ..research.fetcher import fetch_page

    log.debug("web_fetch: url=%s", url)
    content = fetch_page(url)
    if not content:
        return f"Could not extract content from {url}. The page may be blocked, empty, or non-HTML."
    return content


WEB_TOOLS = [web_search, web_fetch]
