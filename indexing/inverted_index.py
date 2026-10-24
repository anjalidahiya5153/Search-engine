"""
Sharded inverted index stored as pickle files.

Shard = doc_id % NUM_SHARDS -> each shard holds its own posting lists.
Posting list entry: {doc_id: term_frequency}

Shard metadata: avg_doc_len, doc_lengths (doc_id -> length)
"""

import os
import math
import pickle

from collections import defaultdict
from typing import Dict, List, Tuple

from config.settings import NUM_SHARDS, INVERTED_INDEX_PATH
from indexing.tokenizer import tokenize


class InvertedIndexShard:

    def __init__(self, shard_id: int):
        self.shard_id = shard_id

        # term -> {doc_id: tf}
        self.postings: Dict[
            str,
            Dict[str, int]
        ] = defaultdict(dict)

        self.doc_lengths: Dict[str, int] = {}

        self.total_docs: int = 0
        self.total_tokens: int = 0

    @property
    def avg_doc_len(self) -> float:
        return self.total_tokens / max(
            self.total_docs,
            1
        )

    def add_document(
        self,
        doc_id: str,
        tokens: List[str]
    ) -> None:

        tf: Dict[str, int] = defaultdict(int)

        for t in tokens:
            tf[t] += 1

        self.doc_lengths[doc_id] = len(tokens)
        self.total_tokens += len(tokens)
        self.total_docs += 1

        for term, freq in tf.items():
            self.postings[term][doc_id] = freq

    def save(self, directory: str) -> None:
        os.makedirs(
            directory,
            exist_ok=True
        )

        path = os.path.join(
            directory,
            f"shard_{self.shard_id}.pkl"
        )

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(
        directory: str,
        shard_id: int
    ) -> "InvertedIndexShard":

        path = os.path.join(
            directory,
            f"shard_{shard_id}.pkl"
        )

        with open(path, "rb") as f:
            return pickle.load(f)


def _shard_id(doc_id: str) -> int:
    return (
        int(doc_id, 16) % NUM_SHARDS
        if all(
            c in "0123456789abcdef-"
            for c in doc_id.replace("-", "")
        )
        else hash(doc_id) % NUM_SHARDS
    )


def index_document(
    doc_id: str,
    text: str
) -> None:
    """Add a single document to the correct shard (load -> update -> save)."""

    sid = _shard_id(doc_id)
    path = INVERTED_INDEX_PATH

    try:
        shard = InvertedIndexShard.load(
            path,
            sid
        )
    except FileNotFoundError:
        shard = InvertedIndexShard(sid)

    tokens = tokenize(text)

    shard.add_document(
        doc_id,
        tokens
    )

    shard.save(path)


def load_all_shards() -> List[InvertedIndexShard]:
    shards = []

    for sid in range(NUM_SHARDS):
        try:
            shards.append(
                InvertedIndexShard.load(
                    INVERTED_INDEX_PATH,
                    sid
                )
            )
        except FileNotFoundError:
            shards.append(
                InvertedIndexShard(sid)
            )

    return shards