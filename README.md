# Syco – RAG chat assistant for psychologists (prototype)

Chat with an AI assistant that answers from a curated library of professional books (PDF), with citations.

```
training/*.pdf ──► extract (PyMuPDF, Hebrew fixes) ──► structural parent/child chunking
               ──► [optional] LLM section context ──► BGE-M3 dense+sparse ──► Qdrant (local)

question ──► hybrid search (dense + sparse, RRF) ──► bge-reranker-v2-m3 ──► parent sections
         ──► Claude (CLI or API) ──► answer with [n] citations
```

## Quick start (Windows)

```
run.cmd                        # create venv + install/check dependencies, build the index if empty, start the chat UI
run.cmd ingest [--force]       # build / update the RAG
run.cmd query "question" [--answer]
run.cmd setup                  # only venv + dependencies
```

`run.cmd` reinstalls dependencies automatically when `requirements.txt` changes or an import fails.

## Manual setup

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env        # optional
```

The first run downloads the BGE-M3 and reranker models (~4 GB) from Hugging Face.

## 1. Build the RAG ("training")

Put PDFs in `training/`, then:

```powershell
python scripts/ingest.py --extract-only   # step 1: check extraction quality in data/extracted/*.md
python scripts/ingest.py                  # step 2: chunk, embed, index (only new/changed PDFs)
python scripts/ingest.py --force          # rebuild everything (e.g. after changing chunking settings)
python scripts/ingest.py --contextualize  # also add LLM-written section context (better retrieval, slower)
```

Always look at `data/extracted/*.md` for a new book first: Hebrew order, headings, and missing (scanned) pages.
PDFs removed from `training/` are removed from the index on the next run.

## 2. Test

```powershell
python scripts/query.py "מהן נקודות תקיעה?"            # retrieval only
python scripts/query.py "מהן נקודות תקיעה?" --answer   # with Claude
streamlit run ui/app.py                                # chat UI (http://localhost:8501)
```

The UI has a **Retrieval only** toggle to inspect what the RAG finds without calling Claude.

> The local Qdrant DB allows one process at a time: stop the UI before running `ingest.py`.

## LLM backend

| `SYCO_LLM_BACKEND` | Auth | Use |
|---|---|---|
| `cli` (default) | Claude Code CLI with your claude.ai login | Personal prototyping only |
| `api` | `ANTHROPIC_API_KEY` | Anything used by other people |

Switching is a single setting in `.env`; nothing else changes.

## Layout

```
rag/config.py         settings (env-overridable)
rag/hebrew.py         normalisation, reversed-text repair
rag/pdf_extract.py    PDF -> headings/paragraphs with page numbers
rag/chunking.py       structural parent/child chunking
rag/contextualize.py  optional LLM context per section
rag/embeddings.py     BGE-M3 + reranker
rag/store.py          Qdrant + parent store
rag/retrieval.py      hybrid search -> rerank -> parents
rag/llm.py            CLI / API backends
rag/chat.py           prompt + answer
scripts/ingest.py     build the index
scripts/query.py      CLI test
ui/app.py             Streamlit chat
data/                 generated (gitignored)
```

## Privacy

Do not enter identifiable patient information while using the `cli` backend.
