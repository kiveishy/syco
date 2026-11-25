"""PDF -> ordered list of structural elements (headings + paragraphs) with page numbers.

Uses PyMuPDF font information to recover the book's structure:
  * body font size = the most common size (weighted by characters)
  * headings      = noticeably larger lines, or short bold standalone lines
  * running headers/footers and page numbers are removed
  * Hebrew stored in visual (reversed) order is detected and repaired
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import pymupdf

from . import hebrew

NUMERIC_NOISE_RE = re.compile(r"\d[\d\s,%\-–]{0,8}|\d+\.\d+")   # chart axis labels, stray page numbers
MARKER_RE = re.compile(r"\(?\s*(?:\d{1,2}|[א-ת])\s*[).]|\(\s*(?:\d{1,2}|[א-ת])\s*\)|[•▪●•●▪]")
PATTERN_HEADING_RE = re.compile(
    r"^(?:(?:פגישה|מפגש)\s+(?:מס['׳]?\s*|מספר\s+)?\d{1,2}\b"
    r"|פרק\s+(?:\d{1,2}|[א-ת]{2,8}י(?:ת)?|[א-ת]{2,8}-[א-ת]{2,8})\b"
    r"|(?:Session|Chapter)\s+\d{1,2}\b)"
)
MIRRORED_MARKER_RE =re.compile(r"^\)\s*(\d{1,2}|[א-ת])\s*\(\s*")
SENT_END_RE =re.compile(r"[.!?:\"”]\s*$")
LIST_ITEM_RE = re.compile(r"^\s*(?:[•●▪–\-\*•▪●]|\d{1,2}[.)]|[א-ת][.)]|\(\s*(?:\d{1,2}|[א-ת])\s*\))\s+")
MARGIN = 0.07           # top/bottom fraction of the page treated as header/footer zone
HEADING_RATIO = 1.12    # size >= body * ratio -> heading candidate


@dataclass
class Line:
    text: str
    size: float
    bold: bool
    y0: float
    y1: float
    page: int  # 1-based


@dataclass
class Element:
    kind: str            # "heading" | "para"
    text: str
    page: int
    level: float = 0     # heading level (1 = top)
    is_list: bool = False


@dataclass
class ExtractResult:
    elements: list[Element]
    num_pages: int
    empty_pages: list[int] = field(default_factory=list)   # likely scanned -> need OCR
    skipped_pages: list[int] = field(default_factory=list)  # TOC / index pages


def _merge_same_baseline(lines: list[dict]) -> list[list[dict]]:
    """PyMuPDF may split one visual line into several 'lines' (e.g. "(", "1", ")", text). Group by vertical overlap."""
    groups: list[tuple[float, float, list[dict]]] = []
    for l in sorted(lines, key=lambda l: l["bbox"][1]):
        y0, y1 = l["bbox"][1], l["bbox"][3]
        mid = (y0 + y1) / 2
        for i, (gy0, gy1, spans) in enumerate(groups):
            if gy0 <= mid <= gy1:
                spans.extend(l["spans"])
                groups[i] = (min(gy0, y0), max(gy1, y1), spans)
                break
        else:
            groups.append((y0, y1, list(l["spans"])))
    return [g[2] for g in groups]


def _line_from_spans(spans: list[dict], page_no: int) -> Line | None:
    """Build a line from character geometry instead of the PDF's internal text order.

    Hebrew PDFs often store punctuation, numbers and brackets in an order that does not match reading order.
    Sorting characters by x gives the *visual* left-to-right string; for an RTL line that is converted to
    logical order by reversing it and re-reversing embedded LTR runs (English, numbers).
    """
    chars = []
    stripped: set[int] = set()  # spans that had leading/trailing spaces removed
    for si, s in enumerate(spans):
        cs = s.get("chars", [])
        # leading/trailing spaces of a span often carry bogus positions and land inside a word
        n0 = len(cs)
        while cs and cs[0]["c"] == " ":
            cs = cs[1:]
        while cs and cs[-1]["c"] == " ":
            cs = cs[:-1]
        if len(cs) != n0:
            stripped.add(si)
        chars += [(c["c"], c["bbox"], s, si) for c in cs]
    if not chars:
        return None
    chars.sort(key=lambda t: (t[1][0] + t[1][2]) / 2)
    # re-insert word gaps: where a span boundary had a (stripped) space, or a clear visual gap
    out: list[str] = []
    prev = None
    for ch, bb, s, si in chars:
        if prev is not None and out and out[-1] != " " and ch != " ":
            p_x1, p_si = prev
            span_gap = si != p_si and (si in stripped or p_si in stripped)
            if span_gap or bb[0] - p_x1 > s["size"] * 0.25:
                out.append(" ")
        out.append(ch)
        prev = (bb[2], si)
    visual = "".join(out)
    rtl = hebrew.hebrew_ratio(visual) > 0.3
    text = hebrew.fix_reversed_line(visual) if rtl else visual
    text = hebrew.normalize(text).strip()
    text = MIRRORED_MARKER_RE.sub(r"(\1) ", text)
    if not text:
        return None
    vis = [s for s in spans if any(c["c"].strip() for c in s.get("chars", []))]
    n = sum(len(s["chars"]) for s in vis)
    size = sum(s["size"] * len(s["chars"]) for s in vis) / max(n, 1)
    bold = all((s["flags"] & 16) or "bold" in s["font"].lower() for s in vis)
    y0 = min(s["bbox"][1] for s in vis)
    y1 = max(s["bbox"][3] for s in vis)
    return Line(text, round(size * 2) / 2, bold, y0, y1, page_no)


def _is_toc_page(lines: list[Line]) -> bool:
    if len(lines) < 8:
        return False
    ends_with_num = sum(1 for ln in lines if re.search(r"(\.{3,}|\s)\d{1,4}$", ln.text) or re.match(r"^\d{1,4}\s", ln.text))
    return ends_with_num / len(lines) > 0.55


def extract(pdf_path: str) -> ExtractResult:
    doc = pymupdf.open(pdf_path)

    # ---- pass 1: collect lines grouped by block -------------------------------
    pages: list[list[list[Line]]] = []   # page -> blocks -> lines
    empty_pages: list[int] = []
    margin_texts: Counter[str] = Counter()
    for pno, page in enumerate(doc, start=1):
        h = page.rect.height
        blocks_out: list[list[Line]] = []
        d = page.get_text("rawdict")
        for b in d["blocks"]:
            if b.get("type") != 0:
                continue
            lines = [ln for spans in _merge_same_baseline(b["lines"]) if (ln := _line_from_spans(spans, pno))]
            lines = [ln for ln in lines if not NUMERIC_NOISE_RE.fullmatch(ln.text)]
            if lines:
                blocks_out.append(lines)
        blocks_out.sort(key=lambda bl: bl[0].y0)
        if sum(len(l.text) for bl in blocks_out for l in bl) < 30:
            empty_pages.append(pno)
        for bl in blocks_out:
            for ln in bl:
                if ln.y1 < h * MARGIN or ln.y0 > h * (1 - MARGIN):
                    margin_texts[re.sub(r"\d+", "#", ln.text)] += 1
        pages.append(blocks_out)

    n = len(doc)
    repeated = {t for t, c in margin_texts.items() if c >= max(3, n * 0.2)}

    def is_noise(ln: Line, h: float) -> bool:
        in_margin = ln.y1 < h * MARGIN or ln.y0 > h * (1 - MARGIN)
        if not in_margin:
            return False
        key = re.sub(r"\d+", "#", ln.text)
        return key in repeated or re.fullmatch(r"[\d\s\-–|.]+", ln.text) is not None

    # ---- pass 2: font statistics ---------------------------------------------
    size_counter: Counter[float] = Counter()
    for pno, blocks in enumerate(pages, start=1):
        h = doc[pno - 1].rect.height
        for bl in blocks:
            for ln in bl:
                if not is_noise(ln, h):
                    size_counter[ln.size] += len(ln.text)
    body = size_counter.most_common(1)[0][0] if size_counter else 10.0
    heading_sizes = sorted({s for s in size_counter if s >= body * HEADING_RATIO}, reverse=True)[:3]
    level_of = {s: i + 1 for i, s in enumerate(heading_sizes)}
    bold_level = len(heading_sizes) + 1

    # ---- pass 3: elements ----------------------------------------------------
    elements: list[Element] = []
    skipped: list[int] = []
    for pno, blocks in enumerate(pages, start=1):
        h = doc[pno - 1].rect.height
        clean_blocks = [[ln for ln in bl if not is_noise(ln, h)] for bl in blocks]
        clean_blocks = [bl for bl in clean_blocks if bl]
        page_lines = [ln for bl in clean_blocks for ln in bl]
        n_pattern = sum(1 for ln in page_lines if len(ln.text) < 90 and PATTERN_HEADING_RE.match(ln.text))
        early = pno <= max(15, n // 10)
        if _is_toc_page(page_lines) or (early and (n_pattern >= 2 or any("תוכן העניינים" in ln.text for ln in page_lines))):
            skipped.append(pno)
            continue
        para: list[str] = []
        last_y1 = None

        def flush() -> None:
            if para:
                txt = _join_lines(para)
                elements.append(Element("para", txt, pno, is_list=bool(LIST_ITEM_RE.match(txt))))
                para.clear()

        marker = ""  # list marker found as its own line, e.g. "(1)"; attached to the next line
        for bl in clean_blocks:
            if len(bl) == 1 and MARKER_RE.fullmatch(bl[0].text):
                flush()
                marker = bl[0].text.replace(" ", "")
                continue
            if marker:
                bl = [Line(f"{marker} {bl[0].text}", bl[0].size, bl[0].bold, bl[0].y0, bl[0].y1, bl[0].page), *bl[1:]]
                marker = ""
            # a new block continues the current paragraph unless the previous line ended a sentence
            # or there is a clear vertical gap (some PDFs put every line in its own block)
            if para and last_y1 is not None:
                gap = bl[0].y0 - last_y1
                if SENT_END_RE.search(para[-1]) or gap > bl[0].size * 1.2 or gap < -bl[0].size:
                    flush()
            for ln in bl:
                lvl = _heading_level(ln, body, level_of, bold_level, single_line_block=len(bl) == 1)
                if lvl:
                    flush()
                    # merge multi-line headings (same level, consecutive)
                    if elements and elements[-1].kind == "heading" and elements[-1].level == lvl and elements[-1].page == pno:
                        elements[-1].text += " " + ln.text
                    else:
                        elements.append(Element("heading", ln.text, pno, level=lvl))
                elif LIST_ITEM_RE.match(ln.text) and para:
                    flush()
                    para.append(ln.text)
                else:
                    para.append(ln.text)
                last_y1 = ln.y1
        flush()

    return ExtractResult(_merge_split_paragraphs(elements), n, empty_pages, skipped)


def _heading_level(ln: Line, body: float, level_of: dict[float, int], bold_level: int, single_line_block: bool) -> float:
    t = ln.text
    if len(t) > 120 or len(t) < 2 or re.fullmatch(r"[\d\W]+", t):
        return 0
    # headings written in body font (e.g. "פגישה מס' 2 – משמעות האירוע"): sit between size-based and bold headings
    if len(t) < 90 and PATTERN_HEADING_RE.match(t):
        return bold_level - 0.5
    if ln.size in level_of:
        return level_of[ln.size]
    if ln.bold and single_line_block and len(t) < 80 and not LIST_ITEM_RE.match(t) and abs(ln.size - body) < 1 and not t.endswith((".", ",", ":")):
        return bold_level
    return 0


def _join_lines(lines: list[str]) -> str:
    out = ""
    for ln in lines:
        if out.endswith("-") and re.search(r"[A-Za-z]-$", out):  # English hyphenation
            out = out[:-1] + ln
        else:
            out = f"{out} {ln}" if out else ln
    return re.sub(r"\s+", " ", out).strip()


def _merge_split_paragraphs(elements: list[Element]) -> list[Element]:
    """Paragraphs broken across a page boundary: join when the first does not end a sentence."""
    out: list[Element] = []
    for el in elements:
        prev = out[-1] if out else None
        if (
            prev and el.kind == "para" and prev.kind == "para" and not el.is_list
            and el.page == prev.page + 1 and not re.search(r"[.!?:\"']$", prev.text)
        ):
            prev.text = f"{prev.text} {el.text}"
        else:
            out.append(el)
    return out


def to_markdown(title: str, res: ExtractResult) -> str:
    """Readable dump for manual inspection of extraction quality."""
    lines = [f"# {title}", "", f"<!-- pages={res.num_pages} "
             f"empty_pages={res.empty_pages} skipped_pages={res.skipped_pages} -->", ""]
    for el in res.elements:
        if el.kind == "heading":
            lines += ["", f"{'#' * min(int(el.level) + 1, 6)} {el.text}  <sub>[p.{el.page}]</sub>", ""]
        else:
            lines += [f"{el.text}  <sub>[p.{el.page}]</sub>", ""]
    return "\n".join(lines)
