"""Build / update the RAG from the PDFs in the training folder.

    python scripts/ingest.py                      # ingest new or changed PDFs in training/ (appends; others untouched)
    python scripts/ingest.py --file book.pdf      # ingest / append only this PDF (copied into training/ if elsewhere)
    python scripts/ingest.py --list               # show indexed books
    python scripts/ingest.py --force              # rebuild everything
    python scripts/ingest.py --extract-only       # only extract to data/extracted/*.md, for checking text quality
    python scripts/ingest.py --contextualize      # also write LLM context per section (slow on the CLI backend)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import config, store  # noqa: E402
from rag.pdf_extract import extract, to_markdown  # noqa: E402


def file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest() -> dict:
    if config.MANIFEST_PATH.exists():
        return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def save_manifest(m: dict) -> None:
    config.MANIFEST_PATH.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def _into_library(src: Path, library: Path, force: bool) -> Path:
    """Keep training/ the single source of truth: a PDF given from elsewhere is copied into it,
    so a later full run does not treat it as removed."""
    if not src.is_file() or src.suffix.lower() != ".pdf":
        sys.exit(f"Not a PDF file: {src}")
    library.mkdir(parents=True, exist_ok=True)
    dest = library / src.name
    if src.resolve() == dest.resolve():
        return dest
    if dest.exists() and file_hash(dest) != file_hash(src) and not force:
        sys.exit(f"{dest} already exists with different content. Rename the file or use --force to replace it.")
    shutil.copy2(src, dest)
    print(f"Copied {src} -> {dest}")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(config.TRAINING_DIR))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--contextualize", action="store_true")
    ap.add_argument("--file", action="append", default=[], metavar="PDF",
                    help="ingest only this PDF (repeatable); appended to the existing index")
    ap.add_argument("--list", action="store_true", help="list indexed books and exit")
    args = ap.parse_args()

    config.ensure_dirs()
    manifest = load_manifest()

    if args.list:
        if not manifest:
            print("No books indexed yet.")
        for name, e in sorted(manifest.items()):
            print(f"{e['title']}\n    file={name} pages={e['pages']} sections={e['parents']} chunks={e['children']} "
                  f"contextualized={e.get('contextualized', False)} ingested={e['ingested_at']}")
        return

    input_dir = Path(args.input)
    if args.file:
        pdfs = [_into_library(Path(f), input_dir, args.force) for f in args.file]
    else:
        pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {input_dir}")
        return

    for pdf in pdfs:
        digest = file_hash(pdf)
        book_id = "b" + digest[:10]
        title = pdf.stem.strip()
        entry = manifest.get(pdf.name)
        if entry and entry["hash"] == digest and not args.force and not args.extract_only and entry.get("indexed"):
            print(f"= {title}: unchanged, skipping")
            continue

        print(f"\n> {title}")
        t0 = time.time()
        res = extract(str(pdf))
        md_path = config.EXTRACTED_DIR / f"{book_id}.md"
        md_path.write_text(to_markdown(title, res), encoding="utf-8")
        n_head = sum(1 for e in res.elements if e.kind == "heading")
        n_para = sum(1 for e in res.elements if e.kind == "para")
        print(f"  pages={res.num_pages} headings={n_head} paragraphs={n_para}")
        if res.empty_pages:
            print(f"  WARNING {len(res.empty_pages)} pages have no text (scanned? needs OCR): {res.empty_pages[:20]}")
        if res.skipped_pages:
            print(f"  skipped TOC/index-like pages: {res.skipped_pages[:20]}")
        print(f"  extracted text -> {md_path}")
        if args.extract_only:
            continue

        from rag import embeddings
        from rag.chunking import chunk_book

        parents, children = chunk_book(book_id, title, res.elements, embeddings.count_tokens)
        print(f"  parents={len(parents)} children={len(children)} "
              f"avg_child_tokens={sum(c.token_count for c in children) // max(len(children), 1)}")

        if args.contextualize:
            from rag.contextualize import contextualize
            from rag.llm import get_llm
            print("  writing section context with the LLM ...")
            contextualize(parents, children, get_llm())

        store.write_jsonl(store.parents_path(book_id), [p.__dict__ for p in parents])
        store.write_jsonl(store.children_path(book_id), [c.to_dict() for c in children])

        print("  embedding ...")
        dense, sparse = embeddings.embed([c.embed_text() for c in children])
        if entry and entry.get("book_id") and entry["book_id"] != book_id:
            store.delete_book(entry["book_id"])
        store.delete_book(book_id)
        store.upsert([c.to_dict() for c in children], dense, sparse)

        manifest[pdf.name] = {
            "hash": digest, "book_id": book_id, "title": title, "pages": res.num_pages,
            "parents": len(parents), "children": len(children), "indexed": True,
            "contextualized": args.contextualize, "ingested_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_manifest(manifest)
        print(f"  done in {time.time() - t0:.0f}s")

    # books removed from the training folder -> remove from index (only on a full-folder run)
    present = {p.name for p in pdfs} if not args.file else set(manifest)
    for name in [n for n in manifest if n not in present]:
        print(f"- {manifest[name]['title']}: removed from folder, deleting from index")
        store.delete_book(manifest[name]["book_id"])
        for p in (store.parents_path(manifest[name]["book_id"]), store.children_path(manifest[name]["book_id"])):
            p.unlink(missing_ok=True)
        del manifest[name]
    save_manifest(manifest)
    if not args.extract_only and store.client.cache_info().currsize:  # only if the DB was touched
        print(f"\nIndex contains {store.count()} chunks.")
    elif not args.extract_only:
        print("\nNothing to do: index is up to date.")
    store.close()


if __name__ == "__main__":
    main()
