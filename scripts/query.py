"""Command-line test of the RAG.

    python scripts/query.py "מהן נקודות תקיעה?"              # retrieval only: show matched passages
    python scripts/query.py "מהן נקודות תקיעה?" --answer     # also ask Claude
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from rag import chat, retrieval  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--answer", action="store_true")
    args = ap.parse_args()

    if not args.answer:
        for s in retrieval.retrieve(args.question):
            print(f"\n[{s.n}] {s.label}  score={s.score:.3f}")
            for m in s.matched:
                print("   ", m[:400].replace("\n", " "), "…" if len(m) > 400 else "")
        return

    turn = chat.ask(args.question, [])
    for piece in turn.answer_stream:
        print(piece, end="", flush=True)
    print("\n\nSources:")
    for s in turn.sources:
        print(f"  [{s.n}] {s.label}")


if __name__ == "__main__":
    try:
        main()
    finally:
        from rag import store
        store.close()
