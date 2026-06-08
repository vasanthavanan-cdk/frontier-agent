"""Fetch and extract main content from web pages.

Uses trafilatura for content extraction — handles article bodies, docs pages,
and blog posts while stripping nav, ads, footers, and other boilerplate.
Falls back to httpx raw text on extraction failure.
"""
from __future__ import annotations

import httpx

from ..logger import get_logger

log = get_logger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# skip binary/media/download URLs
_SKIP_EXTENSIONS = {".pdf", ".zip", ".tar", ".gz", ".png", ".jpg", ".svg", ".mp4"}

_MAX_CONTENT_CHARS = 4000   # cap per page to stay within context window
_MIN_CONTENT_CHARS = 150    # skip pages that return almost nothing useful


def _should_skip(url: str) -> bool:
    lower = url.lower().split("?")[0]
    return any(lower.endswith(ext) for ext in _SKIP_EXTENSIONS)


def fetch_page(url: str, timeout: int = 10) -> str | None:
    """Fetch `url` and return extracted main content, or None on failure."""
    if _should_skip(url):
        log.debug("Skipping non-HTML URL: %s", url)
        return None
    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True, headers=_HEADERS)
        response.raise_for_status()
        html = response.text

        # try trafilatura first (best quality)
        try:
            import trafilatura
            content = trafilatura.extract(
                html,
                include_links=False,
                include_images=False,
                favor_precision=True,
                no_fallback=False,
            )
            if content and len(content) >= _MIN_CONTENT_CHARS:
                return content[:_MAX_CONTENT_CHARS]
        except ImportError:
            pass

        # fallback: strip tags naively
        import re
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) >= _MIN_CONTENT_CHARS:
            return text[:_MAX_CONTENT_CHARS]

    except Exception as e:
        log.debug("Failed to fetch %s: %s", url, e)

    return None


def fetch_all(urls: list[str], max_fetch: int = 6) -> list[dict]:
    """Fetch up to `max_fetch` URLs and return list of {url, title, content}.

    Stops early once enough content is gathered. Skips pages with thin content.
    """
    results: list[dict] = []
    attempted = 0

    for url in urls:
        if len(results) >= max_fetch:
            break
        attempted += 1
        log.debug("Fetching [%d/%d]: %s", attempted, max_fetch, url)
        content = fetch_page(url)
        if content:
            results.append({"url": url, "content": content})
            log.debug("  → extracted %d chars", len(content))
        else:
            log.debug("  → skipped (no content)")

    log.info("Fetched %d/%d pages successfully", len(results), attempted)
    return results
