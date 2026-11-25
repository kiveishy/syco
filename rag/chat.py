"""RAG chat: retrieve sources for the latest question, then ask the LLM to answer from them with citations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from . import retrieval
from .llm import LLM, get_llm
from .retrieval import Source

SYSTEM_PROMPT = """\
You are a clinical knowledge assistant for licensed psychologists. You answer questions using a curated \
library of professional books and documents, which are provided to you as numbered sources in each message.

How to answer:
- Base your answer on the provided sources. Cite them inline with their numbers, e.g. [1] or [2][3], \
right after the claim they support.
- If the sources do not cover the question, or only partly cover it, say so plainly. You may add general \
clinical knowledge, but label it clearly as not coming from the library and do not attach citations to it.
- Answer in the language the psychologist wrote in (Hebrew or English). Keep professional terms \
(e.g. PTSD, CPT, DSM-5) as they appear in the sources.
- Be practical and specific: protocol steps, techniques, wording a therapist can use, what to watch for.
- You support the psychologist's professional judgment; you do not diagnose or replace supervision. \
If a question involves acute risk (suicide, self-harm, harm to others, abuse), remind them of the \
relevant safety and reporting procedures.
"""


@dataclass
class Turn:
    answer_stream: Iterator[str]
    sources: list[Source] = field(default_factory=list)


def _retrieval_query(history: list[dict], question: str) -> str:
    """Short follow-ups ('and in session 4?') need the previous question for context."""
    prev = [m["content"] for m in history if m["role"] == "user"][-1:]
    if prev and len(question.split()) < 8:
        return f"{prev[0]}\n{question}"
    return question


def _user_message(question: str, sources: list[Source]) -> str:
    if not sources:
        src_block = "<sources>\n(no relevant passages were found in the library)\n</sources>"
    else:
        items = [
            f'<source n="{s.n}" ref="{s.label}">\n{s.text}\n</source>' for s in sources
        ]
        src_block = "<sources>\n" + "\n\n".join(items) + "\n</sources>"
    return f"{src_block}\n\n<question>\n{question}\n</question>"


def ask(question: str, history: list[dict], llm: LLM | None = None, book_ids: list[str] | None = None) -> Turn:
    """history = previous turns as [{"role": "user"|"assistant", "content": str}] (plain question/answer text)."""
    llm = llm or get_llm()
    sources = retrieval.retrieve(_retrieval_query(history, question), book_ids)
    # previous turns are sent without their sources to keep the prompt small
    recent = history[-8:]
    while recent and recent[0]["role"] != "user":
        recent = recent[1:]
    messages = [*recent, {"role": "user", "content": _user_message(question, sources)}]
    return Turn(llm.stream(SYSTEM_PROMPT, messages), sources)
