"""
CLI evaluation runner.

Usage:
    python evaluation/run_eval.py --qrels qrels.json --mode hybrid --k 10

qrels.json format:
    [{"query": "...", "relevant": ["doc_id_1", "doc_id_2"]}, ...]
"""

import argparse
import json
import sys

from evaluation.metrics import evaluate
from search import hybrid, bm25, semantic
from config.settings import TOP_K, HYBRID_ALPHA


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--qrels",
        required=True,
        help="Path to qrels JSON file"
    )

    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["hybrid", "bm25", "semantic"]
    )

    parser.add_argument(
        "--k",
        type=int,
        default=TOP_K
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=HYBRID_ALPHA
    )

    args = parser.parse_args()

    with open(args.qrels) as f:
        queries = json.load(f)

    def search_fn(query: str):
        if args.mode == "hybrid":
            results = hybrid.search(
                query,
                top_k=args.k,
                alpha=args.alpha
            )
        elif args.mode == "bm25":
            results = bm25.search(
                query,
                top_k=args.k
            )
        else:
            results = semantic.search(
                query,
                top_k=args.k
            )

        return [r[0] for r in results]

    metrics = evaluate(
        queries,
        search_fn,
        k=args.k
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()