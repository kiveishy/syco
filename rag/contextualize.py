"""Contextual retrieval (optional): one LLM call per parent section writes a short situating description,
which is prepended to every child of that section before embedding.

Per-parent (not per-child) keeps the number of LLM calls manageable on the CLI backend.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from .chunking import Child, Parent
from .llm import LLM

SYSTEM = (
    "You write short retrieval context for passages from professional psychology books. "
    "Reply with only the context sentences, in the same language as the passage, no preamble."
)

PROMPT = """<book>{book}</book>
<location>{path}</location>
<passage>
{text}
</passage>

In 1-2 sentences, state what this passage is about and where it sits in the book (e.g. which protocol, \
session, technique or concept), so that it can be found by search. Mention key clinical terms in both \
Hebrew and English when relevant."""


def contextualize(parents: list[Parent], children: list[Child], llm: LLM, workers: int = 4, log=print) -> None:
    ctx: dict[str, str] = {}

    def one(p: Parent) -> tuple[str, str]:
        res = llm.complete(SYSTEM, [{"role": "user", "content": PROMPT.format(
            book=p.book, path=" > ".join(p.heading_path), text=p.text[:12000])}], max_tokens=400)
        return p.id, ("" if res.error else res.text.strip())

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(one, p) for p in parents]
        for i, f in enumerate(as_completed(futures), 1):
            pid, text = f.result()
            ctx[pid] = text
            if i % 10 == 0 or i == len(parents):
                log(f"    context {i}/{len(parents)}")

    for ch in children:
        ch.context = ctx.get(ch.parent_id, "")
