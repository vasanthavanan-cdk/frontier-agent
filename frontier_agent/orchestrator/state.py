"""Shared state models for the LangGraph orchestration graph.

AgentState is the single source of truth passed between every node in the graph.
All fields use Pydantic v2 so LangGraph can merge partial updates from each node.
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated
import operator

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    """Canonical task categories used by the planner and confidence scorer."""
    coding = "coding"
    research = "research"
    review = "review"
    documentation = "documentation"
    debugging = "debugging"
    planning = "planning"
    general = "general"


class AgentOutput(BaseModel):
    """One agent's response, including the raw content and computed confidence score."""
    agent: str
    content: str
    confidence: float = 0.0
    attempt: int = 1


class TokenUsage(BaseModel):
    """Cumulative token counts and USD cost for one pipeline run."""
    local_tokens: int = 0
    premium_tokens: int = 0
    premium_cost_usd: float = 0.0


class AgentState(BaseModel):
    """Mutable graph state threaded through every node.

    Fields marked with `operator.or_` as their reducer are dict-merged
    by LangGraph when a node returns a partial update, so nodes only need
    to write the keys they touch.
    """
    task_id: str = ""
    original_input: str = ""
    workflow_name: str = "default_coding"
    task_type: TaskType = TaskType.general
    current_step: str = "planner"

    # accumulated outputs — merged across retries
    agent_outputs: Annotated[dict[str, AgentOutput], operator.or_] = Field(default_factory=dict)
    confidence_scores: Annotated[dict[str, float], operator.or_] = Field(default_factory=dict)

    # retry tracking per step
    retry_counts: Annotated[dict[str, int], operator.or_] = Field(default_factory=dict)

    # escalation
    escalation_requested: bool = False
    escalation_approved: bool = False
    escalation_reason: str = ""

    final_output: str = ""
    token_usage: TokenUsage = Field(default_factory=TokenUsage)

    # model tag to use for each agent role (populated from YAML workflow)
    model_assignments: dict[str, str] = Field(default_factory=lambda: {
        "planner": "qwen2.5-coder:7b",
        "coder": "qwen2.5-coder:7b",
        "reviewer": "qwen2.5-coder:7b",
        "documenter": "qwen2.5-coder:7b",
    })

    # configurable thresholds (from YAML workflow)
    confidence_thresholds: dict[str, float] = Field(default_factory=lambda: {
        "planner": 0.70,
        "coder": 0.75,
        "reviewer": 0.80,
        "documenter": 0.65,
        "default": 0.70,
    })

    max_local_retries: int = 2
    fallback_premium_model: str = "claude-sonnet-4-5"

    # Interactive mode (CLI) shows Rich panels and prompts the human before
    # escalating. Headless mode (MCP / Claude-driven) skips the prompt and the
    # premium API call, instead surfacing an escalation recommendation for the
    # caller (Claude) to act on.
    interactive: bool = True

    # web research (populated by researcher_node when research_enabled=True)
    research_enabled: bool = False
    research_max_sources: int = 5
    research_context: str = ""
    research_sources: list[str] = Field(default_factory=list)
    research_queries: list[str] = Field(default_factory=list)
