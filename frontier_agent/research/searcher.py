"""DuckDuckGo web search — returns ranked URLs and snippets, no API key required."""
from __future__ import annotations

from ..logger import get_logger

log = get_logger(__name__)

# source authority tiers — higher = more trusted
_PRIORITY_DOMAINS = [
    "docs.", "documentation.", "developer.", "developers.",  # official docs subdomains
    "github.com", "pypi.org", "npmjs.com",                  # package registries
    "stackoverflow.com", "discuss.",                         # community
    "arxiv.org",                                             # research
]


def _domain_score(url: str) -> int:
    """Return a priority boost for authoritative domains."""
    for i, pattern in enumerate(reversed(_PRIORITY_DOMAINS)):
        if pattern in url:
            return i + 1
    return 0


def search(queries: list[str], max_results_per_query: int = 4) -> list[dict]:
    """Search DuckDuckGo for each query and return deduplicated results sorted by authority.

    Returns list of {url, title, snippet} dicts.
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        log.error("duckduckgo-search not installed — run: pip install duckduckgo-search")
        return []

    seen_urls: set[str] = set()
    results: list[dict] = []

    with DDGS() as ddgs:
        for query in queries:
            log.debug("Searching: %s", query)
            try:
                for r in ddgs.text(query, max_results=max_results_per_query):
                    url = r.get("href", "")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        results.append({
                            "url": url,
                            "title": r.get("title", ""),
                            "snippet": r.get("body", ""),
                            "priority": _domain_score(url),
                        })
            except Exception as e:
                log.warning("Search failed for query '%s': %s", query[:60], e)

    # sort by authority tier descending
    results.sort(key=lambda r: r["priority"], reverse=True)
    log.info("Search found %d unique URLs across %d queries", len(results), len(queries))
    return results
