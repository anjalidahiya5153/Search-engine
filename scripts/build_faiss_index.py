"""
Offline script: read all inverted index shards, collect doc texts from Redis,
embed them, train and persist a fresh FAISS index.

Usage: python scripts/build_faiss_index.py
"""

import json
import logging
import sys

import numpy as np
import redis

from config.settings import REDIS_HOST, REDIS_PORT, REDIS_DB, FAISS_NLIST
from indexing.inverted_index import load_all_shards
from indexing.embedder import embed
from indexing import faiss_index


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BATCH = 512


def main():
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True
    )

    shards = load_all_shards()

    all_doc_ids = []

    for shard in shards:
        all_doc_ids.extend(shard.doc_lengths.keys())

    log.info("Total documents: %d", len(all_doc_ids))

    if not all_doc_ids:
        log.error("No documents found in index. Ingest documents first.")
        sys.exit(1)

    all_vectors = []
    valid_ids = []

    for i in range(0, len(all_doc_ids), BATCH):
        batch_ids = all_doc_ids[i:i + BATCH]

        # Fetch stored text from Redis; key = "doc:<doc_id>"
        pipe = r.pipeline()

        for did in batch_ids:
            pipe.hget(f"doc:{did}", "text")

        texts = pipe.execute()

        filtered_ids, filtered_texts = [], []

        for did, text in zip(batch_ids, texts):
            if text:
                filtered_ids.append(did)
                filtered_texts.append(text)

        if filtered_texts:
            vecs = embed(filtered_texts)
            all_vectors.append(vecs)
            valid_ids.extend(filtered_ids)

        log.info(
            "Embedded %d / %d",
            min(i + BATCH, len(all_doc_ids)),
            len(all_doc_ids)
        )

    vectors = np.vstack(all_vectors).astype(np.float32)

    log.info(
        "Training FAISS index on %d vectors...",
        len(valid_ids)
    )

    faiss_index.train_and_save(vectors, valid_ids)

    log.info("FAISS index saved.")


if __name__ == "__main__":
    main()