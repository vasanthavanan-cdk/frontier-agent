"""Reviewer node: critiques the coder's output for correctness, edge cases, and security.

Expected output format: VERDICT / ISSUES / SUMMARY. Low confidence here (e.g.
NEEDS_REVISION with many issues) can trigger escalation via the router.
"""
from __future__ import annotations

from rich.console import Console

from ..state import AgentState
from .base import call_local_model, research_preamble

console = Console()

import re as _re

_SYSTEM = """You are a critical reviewer. Assess the answer or implementation for quality and accuracy.

Check for and explicitly flag any of these problems:
- DEPRECATED: any API, method, or pattern known to be deprecated or removed in recent versions
- HALLUCINATION: any specific claim (version number, function name, URL, import path) that
  seems invented or contradicts the research context provided above
- INCOMPLETE: TODOs, placeholders, ellipses (...), or missing steps
- UNCERTAIN: vague hedging language that suggests the model is guessing

Respond with:
VERDICT: PASS | NEEDS_REVISION | ESCALATE
(Use ESCALATE when you find DEPRECATED or HALLUCINATION issues that you cannot fix)
ISSUES:
- <DEPRECATED/HALLUCINATION/INCOMPLETE/UNCERTAIN: description> (or "None" if clean)
REVISED_ANSWER:
<improved answer if VERDICT is NEEDS_REVISION; otherwise copy the original unchanged>"""

_REVISED_RE = _re.compile(r"REVISED_ANSWER:\s*\n(.*)", _re.DOTALL | _re.IGNORECASE)
_VERDICT_RE = _re.compile(r"VERDICT:\s*(PASS|NEEDS_REVISION|ESCALATE)", _re.IGNORECASE)


def reviewer_node(state: AgentState) -> AgentState:
    """LangGraph node: review the coder's output and record the verdict in state."""
    console.print("[dim]→ Reviewer checking output...[/dim]")

    coder_out = state.agent_outputs.get("coder")
    user_msg = (
        research_preamble(state)
        + f"Original request: {state.original_input}\n\n"
        + f"Answer to review:\n{coder_out.content if coder_out else 'No answer provided'}"
    )

    output, confidence = call_local_model(
        state=state,
        role="reviewer",
        system_prompt=_SYSTEM,
        user_message=user_msg,
    )

    # parse verdict — ESCALATE forces immediate escalation regardless of score
    verdict_match = _VERDICT_RE.search(output.content)
    verdict = verdict_match.group(1).upper() if verdict_match else "PASS"

    if verdict == "ESCALATE":
        # reviewer explicitly detected hallucination or deprecated APIs — force escalation
        from ...logger import get_logger as _log
        _log(__name__).warning(
            "[%s] REVIEWER verdict=ESCALATE — forcing escalation", state.task_id
        )
        confidence = 0.0   # guaranteed to trigger escalation via router
        output.confidence = 0.0

    # if reviewer produced a revised answer, promote it so _done_node uses it
    revised_match = _REVISED_RE.search(output.content)
    updated_coder_out = coder_out
    if revised_match and verdict == "NEEDS_REVISION" and coder_out:
        revised_content = revised_match.group(1).strip()
        if revised_content and revised_content != coder_out.content:
            from ..state import AgentOutput
            updated_coder_out = AgentOutput(
                agent="coder",
                content=revised_content,
                confidence=coder_out.confidence,
                attempt=coder_out.attempt,
            )

    return AgentState(**{
        **state.model_dump(),
        "current_step": "done",
        "agent_outputs": {
            **state.agent_outputs,
            "coder": updated_coder_out,
            "reviewer": output,
        },
        "confidence_scores": {**state.confidence_scores, "reviewer": confidence},
        "retry_counts": {**state.retry_counts, "reviewer": state.retry_counts.get("reviewer", 0) + 1},
    })
