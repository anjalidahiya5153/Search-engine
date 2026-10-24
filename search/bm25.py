"""BM25 retrieval over sharded inverted index."""
import math
from typing import Dict, List, Tuple

from config.settings import BM25_K1, BM25_B, TOP_K
from indexing.inverted_index import InvertedIndexShard, load_all_shards
from indexing.tokenizer import tokenize


def _bm25_score(
    tf: int,
    df: int,
    total_docs: int,
    doc_len: int,
    avg_doc_len: float,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> float:
    idf = math.log(
        (total_docs - df + 0.5) / (df + 0.5) + 1
    )

    tf_norm = (
        (tf * (k1 + 1))
        / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
    )

    return idf * tf_norm


def search(
    query: str,
    top_k: int = TOP_K,
    shards: List[InvertedIndexShard] | None = None
) -> List[Tuple[str, float]]:
    """Return (doc_id, bm25_score) sorted descending."""

    if shards is None:
        shards = load_all_shards()

    query_tokens = tokenize(query)
    scores: Dict[str, float] = {}

    for shard in shards:
        total_docs = shard.total_docs
        avg_dl = shard.avg_doc_len

        for term in query_tokens:
            if term not in shard.postings:
                continue

            posting = shard.postings[term]
            df = len(posting)

            for doc_id, tf in posting.items():
                doc_len = shard.doc_lengths.get(doc_id, avg_dl)

                s = _bm25_score(
                    tf,
                    df,
                    total_docs,
                    doc_len,
                    avg_dl
                )

                scores[doc_id] = scores.get(doc_id, 0.0) + s

    return sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]