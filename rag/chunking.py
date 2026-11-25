"""Structural, parent-child chunking.

  sections  : split at the book's headings (heading_path kept as metadata)
  parents   : a whole section, or a slice of it, up to PARENT_MAX_TOKENS  -> what the LLM reads
  children  : ~CHILD_TARGET_TOKENS pieces of a parent                     -> what the search indexes
Lists (and the sentence introducing them) are kept in one piece whenever possible.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Callable

from . import config
from .pdf_extract import LIST_ITEM_RE, Element

SENT_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+(?=\S)")
DIALOGUE_RE = re.compile(r"^(?:מטפל|מטפלת|מטופל|מטופלת|המטפל|המטופל|T|C|Therapist|Client|Patient)\s*:", re.M)
WORKSHEET_WORDS = ("דף עבודה", "דפי עבודה", "טופס", "שאלון", "worksheet", "handout", "form")
CASE_WORDS = ("תיאור מקרה", "דוגמה קלינית", "מקרה", "וינייטה", "case example", "vignette", "case study")


@dataclass
class Unit:
    text: str
    page_start: int
    page_end: int
    is_list: bool = False


@dataclass
class Parent:
    id: str
    book_id: str
    book: str
    heading_path: list[str]
    text: str
    page_start: int
    page_end: int
    token_count: int


@dataclass
class Child:
    id: str
    parent_id: str
    book_id: str
    book: str
    heading_path: list[str]
    text: str
    page_start: int
    page_end: int
    token_count: int
    content_type: str = "prose"
    context: str = ""   # optional LLM-written situating sentence (contextual retrieval)
    extra: dict = field(default_factory=dict)

    def embed_text(self) -> str:
        """Text that is embedded/indexed: location + optional context + the chunk itself."""
        loc = " > ".join([self.book, *self.heading_path])
        parts = [loc]
        if self.context:
            parts.append(self.context)
        parts.append(self.text)
        return "\n".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
def _sections(elements: list[Element]) -> list[tuple[list[str], list[Element]]]:
    stack: list[tuple[int, str]] = []
    sections: list[tuple[list[str], list[Element]]] = []
    current: list[Element] = []

    def close() -> None:
        if current:
            sections.append(([t for _, t in stack], list(current)))
            current.clear()

    for el in elements:
        if el.kind == "heading":
            close()
            while stack and stack[-1][0] >= el.level:
                stack.pop()
            stack.append((el.level, el.text))
        else:
            current.append(el)
    close()
    return sections


def _units(paras: list[Element]) -> list[Unit]:
    """Merge consecutive list items (plus the intro sentence ending with ':') into one unit."""
    units: list[Unit] = []
    for p in paras:
        if p.is_list and units and (units[-1].is_list or units[-1].text.rstrip().endswith(":")):
            u = units[-1]
            u.text += "\n" + p.text
            u.page_end = p.page
            u.is_list = True
        else:
            units.append(Unit(p.text, p.page, p.page, p.is_list))
    return units


def _split_long(unit: Unit, max_tokens: int, count: Callable[[str], int]) -> list[Unit]:
    if count(unit.text) <= max_tokens:
        return [unit]
    seps = unit.text.split("\n") if unit.is_list else SENT_SPLIT_RE.split(unit.text)
    joiner = "\n" if unit.is_list else " "
    out: list[Unit] = []
    buf: list[str] = []
    for s in seps:
        if buf and count(joiner.join(buf + [s])) > max_tokens:
            out.append(Unit(joiner.join(buf), unit.page_start, unit.page_end, unit.is_list))
            buf = []
        if count(s) > max_tokens:  # a single monster sentence: hard split by words
            words, piece = s.split(), []
            for w in words:
                if piece and count(" ".join(piece + [w])) > max_tokens:
                    out.append(Unit(" ".join(piece), unit.page_start, unit.page_end))
                    piece = []
                piece.append(w)
            buf = [" ".join(piece)] if piece else []
        else:
            buf.append(s)
    if buf:
        out.append(Unit(joiner.join(buf), unit.page_start, unit.page_end, unit.is_list))
    return out


def _pack(units: list[Unit], target: int, count: Callable[[str], int]) -> list[list[Unit]]:
    groups: list[list[Unit]] = []
    cur: list[Unit] = []
    cur_tokens = 0
    for u in units:
        t = count(u.text)
        if cur and cur_tokens + t > target:
            groups.append(cur)
            cur, cur_tokens = [], 0
        cur.append(u)
        cur_tokens += t
    if cur:
        groups.append(cur)
    # merge a tiny tail into the previous group
    if len(groups) > 1 and sum(count(u.text) for u in groups[-1]) < config.MIN_CHUNK_TOKENS:
        groups[-2].extend(groups.pop())
    return groups


BACK_MATTER = ("הערות", "ביבליוגרפיה", "מקורות", "רשימת מקורות", "רשימה ביבליוגרפית", "מפתח", "אינדקס",
               "תוכן העניינים", "notes", "references", "bibliography", "index", "contents")


def _is_back_matter(path: list[str]) -> bool:
    """Endnotes, bibliography, index, TOC: noise for retrieval."""
    return any(h.strip().lower().rstrip(":") in BACK_MATTER for h in path)


def _content_type(heading_path: list[str], text: str) -> str:
    head = " ".join(heading_path[-2:]).lower()
    if any(w in head for w in WORKSHEET_WORDS):
        return "worksheet"
    if any(w in head for w in CASE_WORDS):
        return "case_example"
    if len(DIALOGUE_RE.findall(text)) >= 2:
        return "dialogue"
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if lines and sum(1 for ln in lines if LIST_ITEM_RE.match(ln)) >= max(2, len(lines) // 2):
        return "list"
    return "prose"


def chunk_book(book_id: str, book: str, elements: list[Element], count: Callable[[str], int]) -> tuple[list[Parent], list[Child]]:
    parents: list[Parent] = []
    children: list[Child] = []
    carry: list[Unit] = []  # text of tiny sections carried forward into the next one

    sections = [(p, e) for p, e in _sections(elements) if not _is_back_matter(p)]
    for si, (path, paras) in enumerate(sections):
        units = carry + _units(paras)
        carry = []
        if not units:
            continue
        if sum(count(u.text) for u in units) < config.MIN_CHUNK_TOKENS and si < len(sections) - 1:
            carry = units
            continue

        sized = [p for u in units for p in _split_long(u, config.PARENT_MAX_TOKENS, count)]
        for pgroup in _pack(sized, config.PARENT_MAX_TOKENS, count):
            pid = f"{book_id}_p{len(parents):04d}"
            ptext = "\n\n".join(u.text for u in pgroup)
            parents.append(Parent(pid, book_id, book, path, ptext, pgroup[0].page_start, pgroup[-1].page_end, count(ptext)))

            # lists may stay whole up to 2x the child max; prose is split at CHILD_MAX
            cunits = [c for u in pgroup for c in _split_long(u, config.CHILD_MAX_TOKENS * (2 if u.is_list else 1), count)]
            for ci, cgroup in enumerate(_pack(cunits, config.CHILD_TARGET_TOKENS, count)):
                ctext = "\n\n".join(u.text for u in cgroup)
                children.append(Child(
                    id=f"{pid}_c{ci:02d}", parent_id=pid, book_id=book_id, book=book, heading_path=path,
                    text=ctext, page_start=cgroup[0].page_start, page_end=cgroup[-1].page_end,
                    token_count=count(ctext), content_type=_content_type(path, ctext),
                ))
    return parents, children
