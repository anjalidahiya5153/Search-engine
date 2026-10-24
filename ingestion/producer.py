"""Kafka producer: reads documents from a source and publishes to raw_documents topic."""
import json
import uuid
from typing import Iterator, Dict, Any

from kafka import KafkaProducer
from kafka.errors import KafkaError

from config.settings import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC_RAW


def _make_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        retries=3,
    )


def publish_documents(
    doc_iter: Iterator[Dict[str, Any]]
) -> int:
    """Publish documents from an iterator to Kafka. Returns count of published docs."""
    producer = _make_producer()
    count = 0

    for doc in doc_iter:
        if "doc_id" not in doc:
            doc["doc_id"] = str(uuid.uuid4())

        future = producer.send(KAFKA_TOPIC_RAW, value=doc)

        try:
            future.get(timeout=10)
            count += 1
        except KafkaError as e:
            print(f"Failed to send doc {doc.get('doc_id')}: {e}")

    producer.flush()
    producer.close()

    return count


def publish_from_json(file_path: str) -> int:
    """Convenience stream a JSONL file (one JSON object per line) into Kafka."""

    def _iter():
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    return publish_documents(_iter())