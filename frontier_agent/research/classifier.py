"""Task intent classifier — routes the two retrieval lanes.

A local 7B model is reliable at *reasoning over context it is given* but
unreliable at *recalling or retrieving current facts* (latest versions, news,
prices, dates). Those two competences want different handling:

  * ``reasoning``      — closed-context work (refactor, explain, plan, generate).
                         No web retrieval needed; the local guard handles it.
  * ``current_facts``  — answer depends on up-to-date external state. Must be
                         retrieved from authoritative sources *first*, then the
                         local model only summarizes what was retrieved.

``classify_intent`` is a fast, deterministic keyword/structural pass — no model
call — so it adds no latency. It is deliberately biased toward ``reasoning``:
current-facts retrieval is the expensive lane, so we only take it on a clear
signal.
"""
from __future__ import annotations

import re

from ..logger import get_logger

log = get_logger(__name__)

# Phrases that strongly imply the answer depends on up-to-date external state.
_CURRENT_FACTS_PATTERNS = re.compile(
    r"\b("
    r"latest|newest|most recent|current(?:ly)?|up[- ]?to[- ]?date|"
    r"release notes?|changelog|change log|"
    r"as of (?:today|now|\d{4})|today|right now|this (?:year|month|week)|"
    r"new(?:est)? version|version \d|"
    r"price|cost of|stock|news|who won|when (?:is|was|did)|"
    r"latest version"
    r")\b",
    re.IGNORECASE,
)

# "version" + a question framing ("what", "which", "latest") is a strong signal.
_VERSION_QUESTION_RE = re.compile(
    r"\b(version|release)\b.*\b(latest|current|newest|what|which|now)\b"
    r"|\b(latest|current|newest|what|which)\b.*\b(version|release)\b",
    re.IGNORECASE,
)


def classify_intent(task: str) -> str:
    """Return ``"current_facts"`` or ``"reasoning"`` for a task description.

    Deterministic and cheap. Defaults to ``"reasoning"`` unless a current-facts
    signal is present, so the expensive retrieval lane is only taken when the
    answer plausibly depends on live external state.
    """
    text = task.strip()
    if _CURRENT_FACTS_PATTERNS.search(text) or _VERSION_QUESTION_RE.search(text):
        log.debug("Intent: current_facts — %s", text[:80])
        return "current_facts"
    log.debug("Intent: reasoning — %s", text[:80])
    return "reasoning"
