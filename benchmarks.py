# src/benchmarks.py
import time
import argparse
import statistics
from .cyborg_client import CyborgClient
from .embed import Embedder
from .utils import read_jsonl
import os

def measure_query_latency(index, queries, embedder: Embedder, k=5):
    latencies = []
    for q in queries:
        q_emb = embedder.model.encode([q])[0].tolist()
        t0 = time.time()
        res = index.query(vector=q_emb, top_k=k)
        t1 = time.time()
        latencies.append(t1 - t0)
    return latencies

def run_benchmark(sample_data_path: str, n_queries=20):
    # load sample records
    records = [r for r in read_jsonl(sample_data_path)]
    # build embeddings
    from .embed import build_embeddings
    items = build_embeddings(records)
    cy = CyborgClient()
    idx = cy.create_encrypted_index()
    # upsert once
    cy.upsert_vectors(idx, items, batch_size=16)
    # build some queries (use contents)
    queries = [r['text'] for r in records][:n_queries]
    embedder = Embedder()
    print("Running queries...")
    latencies = measure_query_latency(idx, queries, embedder)
    print("Query stats (seconds):")
    print(f"  count: {len(latencies)}")
    print(f"  mean: {statistics.mean(latencies):.4f}")
    print(f"  median: {statistics.median(latencies):.4f}")
    print(f"  p95: {sorted(latencies)[int(len(latencies)*0.95)-1]:.4f}")
    return latencies

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample_ehr.jsonl")
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()
    run_benchmark(args.data, n_queries=args.n)
