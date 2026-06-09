"""Shared helper for all local-model agent nodes.

`call_local_model` is the single call-site for Ollama inference. It builds
the ChatOllama client from the workflow's model assignment for the given role,
invokes the model, computes a confidence score, and accumulates local token counts.
"""
from __future__ import annotations

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from ..state import AgentOutput, AgentState, TaskType
from ..confidence import compute_confidence
from ...logger import get_logger

log = get_logger(__name__)


def call_local_model(
    state: AgentState,
    role: str,
    system_prompt: str,
    user_message: str,
) -> tuple[AgentOutput, float]:
    """Call the Ollama model assigned to `role` and return (AgentOutput, confidence score).

    The model tag is resolved from `state.model_assignments[role]`, so swapping
    a model only requires a YAML edit — no code changes.
    """
    model_tag = state.model_assignments.get(role, "qwen2.5-coder:7b")
    attempt = state.retry_counts.get(role, 0) + 1

    log.info("[%s] %s  model=%s  attempt=%d", state.task_id, role.upper(), model_tag, attempt)

    llm = ChatOllama(
        model=model_tag,
        base_url="http://localhost:11434",
        temperature=0.2,
        num_predict=4096,
        num_ctx=16384,
        keep_alive="10m",
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]
    response = llm.invoke(messages)
    content = response.content

    output = AgentOutput(agent=role, content=content, attempt=attempt)
    confidence = compute_confidence(
        output,
        state.task_type,
        attempt,
        grounding_context=state.research_context or None,
    )
    output.confidence = confidence

    threshold = state.confidence_thresholds.get(role, state.confidence_thresholds.get("default", 0.70))
    verdict = "PASS" if confidence >= threshold else "LOW "
    log.info(
        "[%s] %s  confidence=%.3f  threshold=%.2f  [%s]  output_tokens≈%d",
        state.task_id, role.upper(), confidence, threshold, verdict, len(content) // 4,
    )

    # track local tokens (rough estimate)
    state.token_usage.local_tokens += len(content) // 4

    return output, confidence


def research_preamble(state: AgentState) -> str:
    """Return a formatted research context block to prepend to user messages.

    Returns empty string when no research context is available, so callers
    can unconditionally prepend it without any branching.
    """
    if not state.research_context:
        return ""
    sources = "\n".join(f"  - {s}" for s in state.research_sources[:6])
    return (
        f"\n=== RESEARCH BRIEF (live web sources, {len(state.research_sources)} pages) ===\n"
        f"{state.research_context}\n"
        f"Sources consulted:\n{sources}\n"
        f"=== END RESEARCH BRIEF ===\n\n"
    )
