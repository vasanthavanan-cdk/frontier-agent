"""Synthesize raw web content into a structured Research Brief using the local model.

The brief is injected into AgentState so every downstream node (planner,
coder, reviewer) works from the same grounded, sourced context.
"""
from __future__ import annotations

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from ..logger import get_logger

log = get_logger(__name__)

_PROMPT_TMPL = (
    "Read the source content below and write a research brief for the question.\n"
    "Format:\nKEY FACTS:\n- (specific facts, version numbers, dates from sources)\n\n"
    "CURRENT BEST PRACTICES:\n- (recommended patterns)\n\n"
    "IMPORTANT CAUTIONS:\n- (deprecations or breaking changes; write 'None' if none)\n\n"
    "OFFICIAL SOURCES:\n- (URLs)\n\n"
    "Rules: only use facts from the provided sources. Max 300 words.\n\n"
    "QUESTION: {task}\n\nSOURCE CONTENT:\n{content}"
)

# how much source content to pass to the synthesizer per page
_CHARS_PER_SOURCE = 2500
# max total context sent to synthesizer (to stay within the model's window)
_MAX_TOTAL_CHARS = 10_000


def synthesize(task: str, sources: list[dict], model: str = "qwen2.5-coder:7b") -> str:
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
            num_ctx=16384,
            keep_alive="10m",
        )
        response = llm.invoke([
            HumanMessage(content=_PROMPT_TMPL.format(task=task, content=combined)),
        ])
        brief = response.content.strip()
        log.info("Research brief generated: %d chars", len(brief))
        return brief
    except Exception as e:
        log.error("Synthesis failed: %s", e)
        # fall back to returning the raw snippets summary
        fallback_lines = [f"- {s['url']}: {s['content'][:200]}..." for s in sources[:3]]
        return "Raw sources (synthesis failed):\n" + "\n".join(fallback_lines)
