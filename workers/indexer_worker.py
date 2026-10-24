"""
Kafka consumer worker: reads raw documents, builds inverted index shard + FAISS vectors.

Run N parallel workers (each in its own process/container) with distinct consumer group IDs
so Kafka partitions are distributed automatically.
"""

import json
import logging
import os
import signal
import sys

from typing import List, Dict, Any

import numpy as np
from kafka import KafkaConsumer
import redis

from config.settings import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_TOPIC_RAW,
    KAFKA_CONSUMER_GROUP,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB
)

from indexing.inverted_index import index_document
from indexing.embedder import embed
from indexing import faiss_index


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

log = logging.getLogger(__name__)

_redis = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

_BATCH_SIZE = int(
    os.getenv("INDEXER_BATCH_SIZE", "32")
)

_running = True


def _handle_sigterm(sig, frame):
    global _running

    log.info("SIGTERM received, stopping worker")
    _running = False


def _flush_batch(
    batch: List[Dict[str, Any]]
) -> None:

    texts = [d.get("text", "") for d in batch]
    doc_ids = [d["doc_id"] for d in batch]

    # Update inverted index (one doc at a time to respect shard locking)
    # persist raw text so scripts/build_faiss can rebuild FAISS index if needed
    # Save document metadata and text to Redis.
    pipe = _redis.pipeline()
    for doc in batch:
        pipe.hset(
            f"doc:{doc['doc_id']}",
            mapping={
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            },
        )
    pipe.execute()

    # Update the sharded inverted index.
    for doc in batch:
        index_document(doc["doc_id"], doc.get("text", ""))

    # Update FAISS
    vectors = embed(texts).astype(np.float32)

    try:
        faiss_index.add_vectors(
            vectors,
            doc_ids
        )
    except Exception:
        # Index not yet trained - will be trained in bulk by the build_index script
        log.warning(
            "FAISS index not ready, skipping vector add for this batch"
        )

    log.info(
        "Indexed batch of %d documents",
        len(batch)
    )


def run() -> None:
    signal.signal(
        signal.SIGTERM,
        _handle_sigterm
    )

    consumer = KafkaConsumer(
        KAFKA_TOPIC_RAW,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=KAFKA_CONSUMER_GROUP,
        value_deserializer=lambda b: json.loads(
            b.decode("utf-8")
        ),
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        max_poll_records=_BATCH_SIZE,
    )

    batch: List[Dict[str, Any]] = []

    try:
        while _running:
            records = consumer.poll(
                timeout_ms=1000
            )

            for tp, msgs in records.items():
                for msg in msgs:
                    batch.append(msg.value)

                    if len(batch) >= _BATCH_SIZE:
                        _flush_batch(batch)
                        batch.clear()
                        consumer.commit()

            if batch and not records:
                _flush_batch(batch)
                batch.clear()
                consumer.commit()

    finally:
        if batch:
            _flush_batch(batch)
            consumer.commit()

        consumer.close()


if __name__ == "__main__":
    run()