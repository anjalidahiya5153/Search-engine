"""FastAPI route definitions."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional

from config.settings import TOP_K, HYBRID_ALPHA
from search import hybrid, bm25, semantic
from search import cache as result_cache

router = APIRouter()


class SearchResult(BaseModel):
    doc_id: str
    score: float


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: List[SearchResult]
    cached: bool = False


@router.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1),
    top_k: int = Query(TOP_K, ge=1, le=100),
    alpha: float = Query(HYBRID_ALPHA, ge=0.0, le=1.0),
    mode: str = Query("hybrid", regex="^(hybrid|bm25|semantic)$"),
):
    # Check cache
    cached = result_cache.get(q, top_k, alpha)

    if cached:
        return SearchResponse(
            query=q,
            mode=mode,
            results=[SearchResult(doc_id=r[0], score=r[1]) for r in cached],
            cached=True,
        )

    if mode == "hybrid":
        raw = hybrid.search(q, top_k=top_k, alpha=alpha)
    elif mode == "bm25":
        raw = bm25.search(q, top_k=top_k)
    else:
        raw = semantic.search(q, top_k=top_k)

    result_cache.set(q, top_k, alpha, raw)

    return SearchResponse(
        query=q,
        mode=mode,
        results=[SearchResult(doc_id=r[0], score=r[1]) for r in raw],
    )


@router.get("/health")
def health():
    return {"status": "ok"}