from __future__ import annotations

import re
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from .tasks.suite import BenchTask

_JUDGE_MODEL = "gemma4:12b"

_SYSTEM = """You are a strict technical evaluator scoring AI-generated answers.
Score the answer on a scale of 1-5 using ONLY these criteria:
  5 — Complete, correct, production-quality. Covers all edge cases.
  4 — Correct and complete. Minor style issues only.
  3 — Mostly correct. Missing 1-2 notable aspects.
  2 — Partially correct. Significant gaps or errors.
  1 — Incorrect, incomplete, or off-topic.

Respond with EXACTLY this format (no other text):
SCORE: <1-5>
REASON: <one sentence>"""


def judge(task: BenchTask, answer: str) -> tuple[int, str]:
    """Return (score 1-5, one-line reason) using Gemma 4 12B as judge."""
    hint_check = sum(1 for h in task.quality_hints if h.lower() in answer.lower())
    hint_ratio = hint_check / max(len(task.quality_hints), 1)

    llm = ChatOllama(model=_JUDGE_MODEL, base_url="http://localhost:11434", temperature=0.0)
    prompt = (
        f"QUESTION: {task.prompt}\n\n"
        f"ANSWER:\n{answer[:3000]}\n\n"   # cap to avoid context overflow
        f"Quality hints present ({hint_check}/{len(task.quality_hints)}): "
        + ", ".join(h for h in task.quality_hints if h.lower() in answer.lower())
    )
    response = llm.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)])
    text = response.content.strip()

    score_match = re.search(r"SCORE:\s*([1-5])", text)
    reason_match = re.search(r"REASON:\s*(.+)", text)

    score = int(score_match.group(1)) if score_match else max(1, round(hint_ratio * 5))
    reason = reason_match.group(1).strip() if reason_match else "No reason provided."

    return score, reason
