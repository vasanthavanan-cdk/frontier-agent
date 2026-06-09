"""Generate targeted web search queries from a task description using the local model."""
from __future__ import annotations

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from ..logger import get_logger

log = get_logger(__name__)

_PROMPT_TMPL = (
    "Generate 3 Google search queries for the following question. "
    "Output ONLY the queries, one per line, no other text.\n\n"
    "Question: {task}\n\nQueries:"
)


def generate_queries(task: str, model: str = "qwen2.5-coder:7b") -> list[str]:
    """Use the local model to generate 3-4 targeted search queries for the given task."""
    log.debug("Generating search queries for: %s", task[:100])
    try:
        llm = ChatOllama(
            model=model,
            base_url="http://localhost:11434",
            temperature=0.1,
            num_predict=256,
            num_ctx=8192,
            keep_alive="10m",
        )
        response = llm.invoke([
            HumanMessage(content=_PROMPT_TMPL.format(task=task)),
        ])
        queries = [
            q.strip()
            for q in response.content.strip().splitlines()
            if q.strip() and not q.strip().startswith("#")
        ]
        queries = queries[:4]
        log.debug("Generated %d queries: %s", len(queries), queries)
        return queries
    except Exception as e:
        log.warning("Query generation failed: %s — using task as query", e)
        return [task[:120]]
