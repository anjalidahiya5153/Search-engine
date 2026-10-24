"""
Publish documents from a JSONL file to Kafka (default: samples/sample_docs.jsonl).

Usage:
  python scripts/ingest_sample.py [path/to/docs.jsonl]

Each line: {"doc_id": "...", "title": "...", "text": "..."}  (doc_id and title optional)
"""
import sys

from ingestion.producer import publish_from_json

DEFAULT_PATH = "samples/sample_docs.jsonl"

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    print(f"Publishing {path} to Kafka...")
    print(f"Done. Published {publish_from_json(path)} documents.")