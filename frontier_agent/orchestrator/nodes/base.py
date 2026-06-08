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
    model_tag = state.model_assignments.get(role, "gemma4:12b")
    attempt = state.retry_counts.get(role, 0) + 1

    llm = ChatOllama(
        model=model_tag,
        base_url="http://localhost:11434",
        temperature=0.2,
        num_predict=4096,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]
    response = llm.invoke(messages)
    content = response.content

    output = AgentOutput(agent=role, content=content, attempt=attempt)
    confidence = compute_confidence(output, state.task_type, attempt)
    output.confidence = confidence

    # track local tokens (rough estimate)
    state.token_usage.local_tokens += len(content) // 4

    return output, confidence
