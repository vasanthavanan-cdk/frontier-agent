"""Synthesize raw web content into a structured Research Brief using Gemma.

The brief is injected into AgentState so every downstream node (planner,
coder, reviewer) works from the same grounded, sourced context.
"""
from __future__ import annotations

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from ..logger import get_logger

log = get_logger(__name__)

_SYSTEM = """You are a technical research synthesizer. Given a task and raw content from multiple
web sources, produce a concise, structured Research Brief.

Format your output EXACTLY as:

KEY FACTS:
- <fact 1 — specific, verifiable, with version/date if available>
- <fact 2>
...

CURRENT BEST PRACTICES:
- <practice 1>
- <practice 2>
...

IMPORTANT CAUTIONS:
- <deprecation, breaking change, or known gotcha>
(write "None found" if nothing notable)

OFFICIAL SOURCES:
- <URL 1>
- <URL 2>
...

Rules:
- Only include what was actually found in the provided sources
- Prefer specifics (version numbers, API names, method signatures) over generalities
- Flag anything that might be outdated or version-specific
- Max 400 words total"""

# how much source content to pass to the synthesizer per page
_CHARS_PER_SOURCE = 2500
# max total context sent to synthesizer (to stay within Gemma's window)
_MAX_TOTAL_CHARS = 10_000


def synthesize(task: str, sources: list[dict], model: str = "gemma4:12b") -> str:
    """Compress fetched source content into a structured Research Brief.

    Returns the brief as a string, or empty string if no sources were provided.
    """
    if not sources:
        log.info("No sources to synthesize")
        return ""

    # build context from sources, capped to avoid overflowing the model
    blocks: list[str] = []
    total_chars = 0
    for s in sources:
        snippet = s["content"][:_CHARS_PER_SOURCE]
        block = f"SOURCE: {s['url']}\n{snippet}"
        if total_chars + len(block) > _MAX_TOTAL_CHARS:
            break
        blocks.append(block)
        total_chars += len(block)

    combined = "\n\n---\n\n".join(blocks)
    log.debug("Synthesizing %d sources (%d chars) for task: %s", len(blocks), total_chars, task[:80])

    try:
        llm = ChatOllama(
            model=model,
            base_url="http://localhost:11434",
            temperature=0.1,
            num_predict=1024,
        )
        response = llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=f"TASK: {task}\n\nSOURCE CONTENT:\n{combined}"),
        ])
        brief = response.content.strip()
        log.info("Research brief generated: %d chars", len(brief))
        return brief
    except Exception as e:
        log.error("Synthesis failed: %s", e)
        # fall back to returning the raw snippets summary
        fallback_lines = [f"- {s['url']}: {s['content'][:200]}..." for s in sources[:3]]
        return "Raw sources (synthesis failed):\n" + "\n".join(fallback_lines)
