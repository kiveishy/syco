"""BGE-M3 embeddings (dense + sparse/lexical) and the bge-reranker-v2-m3 cross-encoder.

Implemented directly on `transformers` (no FlagEmbedding wrapper):
  dense  = normalised [CLS] vector of the last hidden state
  sparse = relu(sparse_linear(hidden_state)) per token, max-pooled per token id, special tokens dropped
Models load lazily on first use.
"""
from __future__ import annotations

from functools import lru_cache

import torch

from . import config


def _device() -> str:
    if config.DEVICE:
        return config.DEVICE
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config.EMBED_MODEL)


def count_tokens(text: str) -> int:
    return len(tokenizer().encode(text, add_special_tokens=False))


@lru_cache(maxsize=1)
def _embedder():
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel

    dev = _device()
    model = AutoModel.from_pretrained(config.EMBED_MODEL).to(dev).eval()
    if dev.startswith("cuda"):
        model = model.half()
    sparse_linear = torch.nn.Linear(model.config.hidden_size, 1)
    state = torch.load(hf_hub_download(config.EMBED_MODEL, "sparse_linear.pt"), map_location="cpu")
    sparse_linear.load_state_dict(state)
    sparse_linear = sparse_linear.to(dev).to(next(model.parameters()).dtype).eval()
    tok = tokenizer()
    special = {tok.cls_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id}
    return model, sparse_linear, special


@lru_cache(maxsize=1)
def _reranker():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = _device()
    tok = AutoTokenizer.from_pretrained(config.RERANK_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(config.RERANK_MODEL).to(dev).eval()
    if dev.startswith("cuda"):
        model = model.half()
    return tok, model


@torch.inference_mode()
def embed(texts: list[str], batch_size: int = 8, max_length: int = 1024) -> tuple[list[list[float]], list[dict[int, float]]]:
    """Returns (dense vectors, sparse vectors as {token_id: weight})."""
    model, sparse_linear, special = _embedder()
    tok = tokenizer()
    dev = next(model.parameters()).device
    dense_out: list[list[float]] = []
    sparse_out: list[dict[int, float]] = []
    # sort by length so batches have similar padding (much faster on CPU)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    results: dict[int, tuple[list[float], dict[int, float]]] = {}
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(dev)
        hidden = model(**enc).last_hidden_state
        dense = torch.nn.functional.normalize(hidden[:, 0].float(), dim=-1)
        weights = torch.relu(sparse_linear(hidden)).squeeze(-1).float()
        for row, i in enumerate(idx):
            sp: dict[int, float] = {}
            for tid, w in zip(enc["input_ids"][row].tolist(), weights[row].tolist()):
                if tid in special or w <= 0:
                    continue
                if w > sp.get(tid, 0.0):
                    sp[tid] = w
            results[i] = (dense[row].tolist(), sp)
        if len(texts) > 50 and (start // batch_size) % 10 == 0:
            print(f"    embedded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    for i in range(len(texts)):
        dense_out.append(results[i][0])
        sparse_out.append(results[i][1])
    return dense_out, sparse_out


@torch.inference_mode()
def rerank(query: str, passages: list[str], batch_size: int = 8, max_length: int = 1024) -> list[float]:
    if not passages:
        return []
    tok, model = _reranker()
    dev = next(model.parameters()).device
    scores: list[float] = []
    for start in range(0, len(passages), batch_size):
        batch = passages[start:start + batch_size]
        enc = tok([query] * len(batch), batch, padding=True, truncation=True, max_length=max_length,
                  return_tensors="pt").to(dev)
        logits = model(**enc).logits.view(-1).float()
        scores.extend(torch.sigmoid(logits).tolist())
    return scores
