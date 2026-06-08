"""Static registry of known Ollama models validated for Mac Mini M4 Pro 24 GB.

REGISTRY maps Ollama tag → ModelInfo (size, speed, roles, description).
ROLE_RECOMMENDATIONS maps role name → recommended specialist tag for `frontier models pull`.

Only models in this registry are shown in `frontier models status`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """Static metadata for one Ollama model."""
    tag: str
    params_b: float        # billions of parameters
    size_gb: float         # disk size at Q4_K_M
    tok_per_sec: float     # estimated tok/s on M4 Pro
    roles: tuple[str, ...]
    description: str


# Known models validated for Mac Mini M4 Pro 24GB
REGISTRY: dict[str, ModelInfo] = {
    "gemma4:12b": ModelInfo(
        tag="gemma4:12b",
        params_b=12,
        size_gb=7.6,
        tok_per_sec=30,
        roles=("planner", "coder", "reviewer", "documenter"),
        description="Default orchestrator — multimodal, native audio/vision, agentic workflows",
    ),
    "qwen2.5-coder:14b": ModelInfo(
        tag="qwen2.5-coder:14b",
        params_b=14,
        size_gb=9.0,
        tok_per_sec=25,
        roles=("coder",),
        description="Coding specialist — pull if Gemma 4 coding quality is insufficient",
    ),
    "deepseek-r1:14b": ModelInfo(
        tag="deepseek-r1:14b",
        params_b=14,
        size_gb=9.0,
        tok_per_sec=22,
        roles=("planner", "reviewer"),
        description="Reasoning specialist — pull if complex multi-step reasoning tasks fail",
    ),
    "qwen3:14b": ModelInfo(
        tag="qwen3:14b",
        params_b=14,
        size_gb=9.0,
        tok_per_sec=24,
        roles=("planner",),
        description="Fallback orchestrator — reliable tool calling if Gemma 4 disappoints",
    ),
}

ROLE_RECOMMENDATIONS: dict[str, str] = {
    "coding":      "qwen2.5-coder:14b",
    "reasoning":   "deepseek-r1:14b",
    "orchestrator": "qwen3:14b",
}


def get(tag: str) -> ModelInfo | None:
    """Return ModelInfo for `tag`, or None if not in the registry."""
    return REGISTRY.get(tag)


def all_models() -> list[ModelInfo]:
    """Return all registered models as a flat list."""
    return list(REGISTRY.values())
