"""Central configuration. Every value can be overridden with an environment variable (or a .env file)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# --- Paths -------------------------------------------------------------------
TRAINING_DIR = Path(_env("SYCO_TRAINING_DIR", str(ROOT / "training")))
DATA_DIR = Path(_env("SYCO_DATA_DIR", str(ROOT / "data")))
EXTRACTED_DIR = DATA_DIR / "extracted"   # human-readable Markdown per book, for inspection
CHUNKS_DIR = DATA_DIR / "chunks"         # parents/children JSONL per book
QDRANT_DIR = DATA_DIR / "qdrant"         # local vector DB
MANIFEST_PATH = DATA_DIR / "manifest.json"

# --- Chunking (token counts use the embedding model's tokenizer) -------------
CHILD_TARGET_TOKENS = int(_env("SYCO_CHILD_TARGET_TOKENS", "320"))
CHILD_MAX_TOKENS = int(_env("SYCO_CHILD_MAX_TOKENS", "450"))
PARENT_MAX_TOKENS = int(_env("SYCO_PARENT_MAX_TOKENS", "1800"))
MIN_CHUNK_TOKENS = int(_env("SYCO_MIN_CHUNK_TOKENS", "40"))

# --- Models ------------------------------------------------------------------
EMBED_MODEL = _env("SYCO_EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = _env("SYCO_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
USE_RERANKER = _env("SYCO_USE_RERANKER", "1") == "1"
DEVICE = _env("SYCO_DEVICE", "")  # "" = auto (cuda if available, else cpu)

# --- Retrieval ---------------------------------------------------------------
COLLECTION = _env("SYCO_COLLECTION", "syco_chunks")
CANDIDATES = int(_env("SYCO_CANDIDATES", "30"))     # hybrid-search hits before reranking
TOP_K = int(_env("SYCO_TOP_K", "6"))                # children kept after reranking
MAX_CONTEXT_TOKENS = int(_env("SYCO_MAX_CONTEXT_TOKENS", "9000"))  # parent text sent to the LLM

# --- LLM ---------------------------------------------------------------------
# "cli" = Claude Code CLI with your claude.ai login (personal prototyping only)
# "api" = Anthropic API with ANTHROPIC_API_KEY (required for anything used by others)
LLM_BACKEND = _env("SYCO_LLM_BACKEND", "cli")
LLM_MODEL = _env("SYCO_LLM_MODEL", "claude-opus-5-5")
LLM_EFFORT = _env("SYCO_LLM_EFFORT", "medium")
CLAUDE_CLI = _env("SYCO_CLAUDE_CLI", "")  # path to claude(.exe); auto-detected if empty
LLM_TIMEOUT_S = int(_env("SYCO_LLM_TIMEOUT_S", "300"))


def ensure_dirs() -> None:
    for d in (DATA_DIR, EXTRACTED_DIR, CHUNKS_DIR, QDRANT_DIR):
        d.mkdir(parents=True, exist_ok=True)
