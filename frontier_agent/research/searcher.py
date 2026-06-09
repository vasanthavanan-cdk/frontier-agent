"""DuckDuckGo web search — returns ranked URLs and snippets, no API key required."""
from __future__ import annotations

import re

from ..logger import get_logger

log = get_logger(__name__)

# source authority tiers — higher = more trusted
_PRIORITY_DOMAINS = [
    "docs.", "documentation.", "developer.", "developers.",  # official docs subdomains
    "github.com", "pypi.org", "npmjs.com",                  # package registries
    "stackoverflow.com", "discuss.",                         # community
    "arxiv.org",                                             # research
]

# Version-pinned CDN URLs (unpkg.com/pkg@2.5.11, cdn.jsdelivr.net/npm/pkg@1.2.3)
# encode a *specific old release* in the path — they are stale by construction and
# poison "what is the latest version" answers. Drop them before they reach synthesis.
_PINNED_VERSION_RE = re.compile(r"@\d+\.\d+(?:\.\d+)?")


def _is_stale_pinned(url: str) -> bool:
    """True for version-pinned package-CDN URLs that snapshot one old release."""
    return bool(_PINNED_VERSION_RE.search(url))


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
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            log.error("ddgs not installed — run: pip install ddgs")
            return []

    seen_urls: set[str] = set()
    results: list[dict] = []

    with DDGS() as ddgs:
        for query in queries:
            log.debug("Searching: %s", query)
            try:
                for r in ddgs.text(query, max_results=max_results_per_query):
                    url = r.get("href", "")
                    if url and _is_stale_pinned(url):
                        log.debug("Dropping version-pinned (stale) URL: %s", url)
                        continue
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


def _tavily_search(queries: list[str], max_results_per_query: int) -> list[dict]:
    """Query the Tavily API for each query, returning results with pre-extracted content.

    Tavily returns clean, recency-ranked results with the page text already
    extracted, so the caller can skip HTML fetching entirely. Each result dict
    carries a non-empty ``content`` field (the extracted page text).
    """
    import httpx

    from ..config import settings

    seen_urls: set[str] = set()
    results: list[dict] = []

    with httpx.Client(timeout=20.0) as client:
        for query in queries:
            log.debug("Tavily searching: %s", query)
            resp = client.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
                json={
                    "api_key": settings.tavily_api_key,
                    "query": query,
                    "max_results": max_results_per_query,
                    "search_depth": "basic",
                    "include_raw_content": True,
                },
            )
            resp.raise_for_status()
            for r in resp.json().get("results", []):
                url = r.get("url", "")
                if not url or url in seen_urls or _is_stale_pinned(url):
                    continue
                seen_urls.add(url)
                content = r.get("raw_content") or r.get("content") or ""
                results.append({
                    "url": url,
                    "title": r.get("title", ""),
                    "snippet": r.get("content", ""),
                    "content": content,
                    "priority": _domain_score(url),
                    "score": float(r.get("score", 0.0)),
                })

    # rank by Tavily relevance score, then domain authority
    results.sort(key=lambda r: (r["score"], r["priority"]), reverse=True)
    log.info("Tavily found %d results across %d queries", len(results), len(queries))
    return results


def search_current_facts(queries: list[str], max_results_per_query: int = 4) -> list[dict]:
    """Retrieval for current-facts queries (latest version, news, prices).

    Uses Tavily when ``TAVILY_API_KEY`` is configured — clean, recency-ranked,
    pre-extracted content, which is exactly what DuckDuckGo + HTML scraping fails
    to provide for "what is the latest X" questions. Falls back to DuckDuckGo
    (with stale-URL filtering) when no key is set or Tavily errors.
    """
    from ..config import settings

    if settings.tavily_api_key:
        try:
            return _tavily_search(queries, max_results_per_query)
        except Exception as e:
            log.warning("Tavily search failed (%s) — falling back to DuckDuckGo", e)
    return search(queries, max_results_per_query)
