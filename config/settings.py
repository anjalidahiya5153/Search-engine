import os

# Kafka

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC_RAW = "raw_documents"
KAFKA_TOPIC_INDEXED = "indexed_documents"
KAFKA_CONSUMER_GROUP = "indexer_group"
KAFKA_NUM_WORKERS = int(os.getenv("KAFKA_NUM_WORKERS", "4"))


# Redis

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_CACHE_TTL = int(os.getenv("REDIS_CACHE_TTL", "3600"))  # seconds


# FAISS / Embeddings

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "/data/faiss_index")
FAISS_NLIST = int(os.getenv("FAISS_NLIST", "100"))  # IVF clusters


# BM25

BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
BM25_B = float(os.getenv("BM25_B", "0.75"))


# Hybrid ranking

HYBRID_ALPHA = float(
    os.getenv("HYBRID_ALPHA", "0.5")
)  # Weight for BM25; (1-alpha) for semantic


# Index sharding

NUM_SHARDS = int(os.getenv("NUM_SHARDS", "4"))
INVERTED_INDEX_PATH = os.getenv(
    "INVERTED_INDEX_PATH", "/data/inverted_index"
)


# API

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
TOP_K = int(os.getenv("TOP_K", "10"))