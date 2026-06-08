"""Pydantic v2 schema for YAML workflow definitions.

A WorkflowSpec is loaded from a .yaml file by workflows/loader.py and then
applied to AgentState before the graph runs. Editing the YAML is the only
way to change model assignments or thresholds — no code restart required.

Invariant: EscalationConfig.require_human_approval cannot be False.
"""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class ModelAssignments(BaseModel):
    """Maps each agent role to an Ollama model tag. Change the tag to swap a model."""
    planner: str = "gemma4:12b"
    coder: str = "gemma4:12b"
    reviewer: str = "gemma4:12b"
    documenter: str = "gemma4:12b"
    fallback_premium: str = "claude-sonnet-4-5"


class ConfidenceThresholds(BaseModel):
    """Minimum confidence score required at each step before the router escalates."""
    planner: float = Field(0.70, ge=0.0, le=1.0)
    coder: float = Field(0.75, ge=0.0, le=1.0)
    reviewer: float = Field(0.80, ge=0.0, le=1.0)
    documenter: float = Field(0.65, ge=0.0, le=1.0)
    default: float = Field(0.70, ge=0.0, le=1.0)


class EscalationConfig(BaseModel):
    """Escalation policy. `require_human_approval` is validated to always be True."""
    require_human_approval: bool = True
    max_local_retries: int = Field(2, ge=1, le=5)

    @field_validator("require_human_approval")
    @classmethod
    def approval_must_be_enabled(cls, v: bool) -> bool:
        if not v:
            raise ValueError(
                "require_human_approval cannot be set to false — "
                "human approval before premium model use is a system invariant."
            )
        return v


class WorkflowStep(BaseModel):
    """One step in the workflow sequence. `depends_on` must reference valid step IDs."""
    id: str
    agent: Literal["planner", "coder", "reviewer", "documenter"]
    input_template: str = "{{user_task}}"
    output_key: str = ""
    depends_on: list[str] = Field(default_factory=list)
    optional: bool = False

    @model_validator(mode="after")
    def set_output_key(self) -> WorkflowStep:
        if not self.output_key:
            self.output_key = self.id
        return self


class ResearchConfig(BaseModel):
    """Controls the web research node that runs before the planner."""
    enabled: bool = False
    max_sources: int = Field(5, ge=1, le=15)


class RoutingRule(BaseModel):
    """Optional declarative routing override (not yet wired into the graph; reserved for future use)."""
    condition: str  # e.g. "confidence['coder'] < thresholds['coder'] and retries >= 2"
    action: Literal["retry", "escalate", "skip"]
    max_retries: int = 2


class WorkflowSpec(BaseModel):
    """Complete validated workflow loaded from a YAML file."""
    version: str = "1.0"
    name: str
    description: str = ""
    models: ModelAssignments = Field(default_factory=ModelAssignments)
    confidence_thresholds: ConfidenceThresholds = Field(default_factory=ConfidenceThresholds)
    escalation: EscalationConfig = Field(default_factory=EscalationConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    steps: list[WorkflowStep] = Field(default_factory=list)
    routing_rules: list[RoutingRule] = Field(default_factory=list)

    @field_validator("steps")
    @classmethod
    def validate_step_deps(cls, steps: list[WorkflowStep]) -> list[WorkflowStep]:
        step_ids = {s.id for s in steps}
        for step in steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    raise ValueError(f"Step '{step.id}' depends on unknown step '{dep}'")
        return steps
