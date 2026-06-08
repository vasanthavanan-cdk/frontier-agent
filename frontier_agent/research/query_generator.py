"""Generate targeted web search queries from a task description using Gemma."""
from __future__ import annotations

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from ..logger import get_logger

log = get_logger(__name__)

_SYSTEM = """You are a research query specialist. Given a task or question, generate 3-4 targeted
web search queries to find the most current and authoritative information.

Focus on:
- Official documentation and changelogs
- Best practices and patterns (2024/2025)
- Known issues, deprecations, migration guides
- Authoritative tutorials from primary sources

Rules:
- Output ONLY the queries, one per line
- No numbering, bullets, or explanation
- Each query should target a different angle (docs, examples, best practices, pitfalls)
- Include version numbers or years when relevant to the task"""


def generate_queries(task: str, model: str = "gemma4:12b") -> list[str]:
    """Use Gemma to generate 3-4 targeted search queries for the given task."""
    log.debug("Generating search queries for: %s", task[:100])
    try:
        llm = ChatOllama(
            model=model,
            base_url="http://localhost:11434",
            temperature=0.2,
            num_predict=256,
        )
        response = llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=task),
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
