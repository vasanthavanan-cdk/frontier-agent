from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from .schema import WorkflowSpec
from ..orchestrator.state import AgentState

_BUILTIN_DIR = Path(__file__).parent / "definitions"


def load_workflow(name: str) -> WorkflowSpec:
    """Load a workflow by name from the definitions directory.

    Searches: definitions/<name>.yaml → definitions/default_<name>.yaml
    Raises FileNotFoundError or ValidationError with clear messages.
    """
    candidates = [
        _BUILTIN_DIR / f"{name}.yaml",
        _BUILTIN_DIR / f"default_{name}.yaml",
    ]
    for path in candidates:
        if path.exists():
            return _parse(path)

    available = [p.stem for p in _BUILTIN_DIR.glob("*.yaml")]
    raise FileNotFoundError(
        f"Workflow '{name}' not found. Available workflows: {available}"
    )


def load_workflow_file(path: Path | str) -> WorkflowSpec:
    """Load a workflow from an explicit file path."""
    return _parse(Path(path))


def _parse(path: Path) -> WorkflowSpec:
    with path.open() as f:
        data = yaml.safe_load(f)
    try:
        return WorkflowSpec.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Invalid workflow '{path}': {e}") from e


def apply_workflow(state: AgentState, workflow: WorkflowSpec) -> AgentState:
    """Overlay workflow config onto an AgentState."""
    m = workflow.models
    t = workflow.confidence_thresholds
    e = workflow.escalation

    return AgentState(**{
        **state.model_dump(),
        "workflow_name": workflow.name,
        "model_assignments": {
            "planner": m.planner,
            "coder": m.coder,
            "reviewer": m.reviewer,
            "documenter": m.documenter,
        },
        "confidence_thresholds": {
            "planner": t.planner,
            "coder": t.coder,
            "reviewer": t.reviewer,
            "documenter": t.documenter,
            "default": t.default,
        },
        "max_local_retries": e.max_local_retries,
        "fallback_premium_model": m.fallback_premium,
    })
