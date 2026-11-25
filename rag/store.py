"""Qdrant vector store (embedded local mode) + parent store (JSONL files).

Children (small chunks) are indexed with a dense and a sparse vector; hybrid search fuses both with RRF.
Parents (the larger sections handed to the LLM) are kept in data/chunks/<book_id>.parents.jsonl.
"""
from __future__ import annotations

import json
import uuid
from functools import lru_cache
from pathlib import Path

from qdrant_client import QdrantClient, models

from . import config

DENSE_DIM = 1024  # BGE-M3


@lru_cache(maxsize=1)
def client() -> QdrantClient:
    config.ensure_dirs()
    return QdrantClient(path=str(config.QDRANT_DIR))


def close() -> None:
    if client.cache_info().currsize:
        client().close()
        client.cache_clear()


def ensure_collection() -> None:
    c = client()
    if c.collection_exists(config.COLLECTION):
        return
    c.create_collection(
        config.COLLECTION,
        vectors_config={"dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )


def _point_id(child_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, child_id))


def delete_book(book_id: str) -> None:
    c = client()
    if not c.collection_exists(config.COLLECTION):
        return
    c.delete(
        config.COLLECTION,
        points_selector=models.FilterSelector(filter=models.Filter(
            must=[models.FieldCondition(key="book_id", match=models.MatchValue(value=book_id))]
        )),
    )


def upsert(children: list[dict], dense: list[list[float]], sparse: list[dict[int, float]]) -> None:
    ensure_collection()
    points = [
        models.PointStruct(
            id=_point_id(ch["id"]),
            vector={
                "dense": d,
                "sparse": models.SparseVector(indices=list(s.keys()), values=list(s.values())),
            },
            payload=ch,
        )
        for ch, d, s in zip(children, dense, sparse)
    ]
    for i in range(0, len(points), 128):
        client().upsert(config.COLLECTION, points=points[i:i + 128])


def hybrid_search(dense: list[float], sparse: dict[int, float], limit: int, book_ids: list[str] | None = None) -> list[dict]:
    flt = None
    if book_ids:
        flt = models.Filter(must=[models.FieldCondition(key="book_id", match=models.MatchAny(any=book_ids))])
    res = client().query_points(
        config.COLLECTION,
        prefetch=[
            models.Prefetch(query=dense, using="dense", limit=limit, filter=flt),
            models.Prefetch(
                query=models.SparseVector(indices=list(sparse.keys()), values=list(sparse.values())),
                using="sparse", limit=limit, filter=flt,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )
    return [{**p.payload, "_score": p.score} for p in res.points]


def count() -> int:
    c = client()
    if not c.collection_exists(config.COLLECTION):
        return 0
    return c.count(config.COLLECTION).count


# ---- parents -----------------------------------------------------------------
def parents_path(book_id: str) -> Path:
    return config.CHUNKS_DIR / f"{book_id}.parents.jsonl"


def children_path(book_id: str) -> Path:
    return config.CHUNKS_DIR / f"{book_id}.children.jsonl"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_parents() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in config.CHUNKS_DIR.glob("*.parents.jsonl"):
        for row in read_jsonl(p):
            out[row["id"]] = row
    return out
