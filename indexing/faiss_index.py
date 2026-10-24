"""
FAISS IVFFlat index for approximate nearest-neighbour semantic search.

doc_ids are stored in a parallel list; FAISS integer IDs map 1-to-1 by position.
"""

import os
import json
from typing import List, Tuple

import faiss
import numpy as np

from config.settings import (
    EMBEDDING_DIM,
    FAISS_NLIST,
    FAISS_INDEX_PATH
)


_INDEX_FILE = os.path.join(
    FAISS_INDEX_PATH,
    "faiss.index"
)

_IDS_FILE = os.path.join(
    FAISS_INDEX_PATH,
    "doc_ids.json"
)


def _build_index(nlist: int) -> faiss.IndexIVFFlat:
    quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)

    index = faiss.IndexIVFFlat(
        quantizer,
        EMBEDDING_DIM,
        nlist,
        faiss.METRIC_INNER_PRODUCT
    )

    return index


def train_and_save(
    vectors: np.ndarray,
    doc_ids: List[str]
) -> None:
    """Train IVF index on vectors (N, DIM) and persist to disk."""

    os.makedirs(
        FAISS_INDEX_PATH,
        exist_ok=True
    )

    nlist = min(
        FAISS_NLIST,
        max(1, len(doc_ids) // 10)
    )

    index = _build_index(nlist)

    index.train(vectors)
    index.add(vectors)

    faiss.write_index(
        index,
        _INDEX_FILE
    )

    with open(_IDS_FILE, "w") as f:
        json.dump(doc_ids, f)


def add_vectors(
    vectors: np.ndarray,
    doc_ids: List[str]
) -> None:
    """Incrementally add vectors to an existing (trained) index."""

    index, existing_ids = load()

    index.add(vectors)
    existing_ids.extend(doc_ids)

    faiss.write_index(
        index,
        _INDEX_FILE
    )

    with open(_IDS_FILE, "w") as f:
        json.dump(existing_ids, f)


def load() -> Tuple[faiss.IndexIVFFlat, List[str]]:
    index = faiss.read_index(_INDEX_FILE)

    with open(_IDS_FILE) as f:
        doc_ids = json.load(f)

    return index, doc_ids


def search(
    query_vec: np.ndarray,
    top_k: int,
    nprobe: int = 10
) -> List[Tuple[str, float]]:
    """Return list of (doc_id, score) sorted by descending score."""

    index, doc_ids = load()

    index.nprobe = nprobe

    query = query_vec.reshape(
        1, -1
    ).astype(np.float32)

    distances, indices = index.search(
        query,
        top_k
    )

    results = []

    for dist, idx in zip(
        distances[0],
        indices[0]
    ):
        if idx == -1:
            continue

        results.append(
            (
                doc_ids[idx],
                float(dist)
            )
        )

    return results