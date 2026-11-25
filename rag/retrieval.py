"""Query -> hybrid search over children -> rerank -> expand to parents (deduplicated, token-budgeted)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from . import config, embeddings, store


@dataclass
class Source:
    n: int                 # citation number shown to the LLM and the user
    parent_id: str
    book: str
    heading_path: list[str]
    page_start: int
    page_end: int
    text: str              # parent text (what the LLM reads)
    matched: list[str]     # child texts that matched (for display)
    score: float

    @property
    def label(self) -> str:
        pages = f"p.{self.page_start}" if self.page_start == self.page_end else f"pp.{self.page_start}-{self.page_end}"
        path = " > ".join(self.heading_path[-2:])
        return f"{self.book} — {path} ({pages})" if path else f"{self.book} ({pages})"


@lru_cache(maxsize=1)
def _parents() -> dict[str, dict]:
    return store.load_parents()


def reload() -> None:
    _parents.cache_clear()


def search_children(query: str, book_ids: list[str] | None = None) -> list[dict]:
    dense, sparse = embeddings.embed([query])
    hits = store.hybrid_search(dense[0], sparse[0], config.CANDIDATES, book_ids)
    if config.USE_RERANKER and hits:
        scores = embeddings.rerank(query, [h["text"] for h in hits])
        for h, s in zip(hits, scores):
            h["_score"] = s
        hits.sort(key=lambda h: h["_score"], reverse=True)
    return hits[: config.TOP_K]


def retrieve(query: str, book_ids: list[str] | None = None) -> list[Source]:
    hits = search_children(query, book_ids)
    parents = _parents()
    sources: list[Source] = []
    by_parent: dict[str, Source] = {}
    budget = config.MAX_CONTEXT_TOKENS
    for h in hits:
        pid = h["parent_id"]
        if pid in by_parent:
            by_parent[pid].matched.append(h["text"])
            continue
        p = parents.get(pid)
        text = p["text"] if p else h["text"]
        tokens = p["token_count"] if p else h["token_count"]
        if tokens > budget:
            # parent too big for the remaining budget: fall back to the matched child only
            text, tokens = h["text"], h["token_count"]
            if tokens > budget:
                continue
        budget -= tokens
        src = Source(
            n=len(sources) + 1, parent_id=pid, book=h["book"], heading_path=h["heading_path"],
            page_start=(p or h)["page_start"], page_end=(p or h)["page_end"],
            text=text, matched=[h["text"]], score=h["_score"],
        )
        sources.append(src)
        by_parent[pid] = src
    return sources
