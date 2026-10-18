"""
Orchestrates the full benchmark: for every data scale, for every enabled
system, build the index once, then sweep ef_search and measure recall,
latency, and throughput at each point. Writes one row per
(region, scale, system, ef_search) combination to results/raw_results.csv.

Every system is driven through the exact same sequence of calls (see
harness/base.py), with the exact same HNSW M / ef_construction, on the same
query set and the same ground truth -- so differences in the output are
attributable to the systems themselves, not to how they were tested.

Usage:
    python run_benchmark.py                  # full run, sized by config.yaml
    python run_benchmark.py --smoke-test       # tiny run against whatever
                                                # data_prep.py --smoke-test produced
    python run_benchmark.py --systems faiss     # only run specific systems
    python run_benchmark.py --region ap-south-1  # tag rows as measured in Mumbai
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import yaml

from harness.faiss_store import FaissStore
from harness.ground_truth import load_or_compute_ground_truth, recall_at_k


def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def build_store(system_name: str, config: dict):
    if system_name == "faiss":
        return FaissStore()
    if system_name == "pgvector":
        from harness.pgvector_store import PgVectorStore
        pg = config["pgvector"]
        dsn = f"host={pg['host']} port={pg['port']} dbname={pg['dbname']} user={pg['user']} password={pg['password']}"
        return PgVectorStore(dsn=dsn, table=pg["table"])
    if system_name == "qdrant":
        from harness.qdrant_store import QdrantStore
        qd = config["qdrant"]
        return QdrantStore(host=qd["host"], port=qd["port"], collection=qd["collection"])
    raise ValueError(f"Unknown system: {system_name}")


def run_one_system_one_scale(system_name, config, region, scale_name, passage_ids, passage_vectors,
                              query_ids, query_vectors, ground_truth, k):
    hnsw = config["hnsw"]
    store = build_store(system_name, config)
    rows = []
    try:
        build_time = store.build_index(passage_ids, passage_vectors, hnsw["M"], hnsw["ef_construction"])
        memory_bytes = store.index_memory_bytes()
        print(f"    [{system_name}] index built in {build_time:.2f}s, "
              f"~{memory_bytes / 1e6:.1f} MB")

        for ef_search in hnsw["ef_search_sweep"]:
            store.set_ef_search(ef_search)
            all_ids, latencies = store.search_many(query_vectors, k)

            recalls = [
                recall_at_k(all_ids[i], ground_truth[i], passage_ids, k)
                for i in range(len(query_ids))
            ]
            latencies_arr = np.array(latencies)
            mean_recall = float(np.mean(recalls))
            qps = 1.0 / float(np.mean(latencies_arr)) if np.mean(latencies_arr) > 0 else 0.0

            rows.append({
                "region": region,
                "scale": scale_name,
                "system": system_name,
                "ef_search": ef_search,
                "recall": mean_recall,
                "qps": qps,
                "p50_latency_ms": float(np.percentile(latencies_arr, 50)) * 1000,
                "p95_latency_ms": float(np.percentile(latencies_arr, 95)) * 1000,
                "p99_latency_ms": float(np.percentile(latencies_arr, 99)) * 1000,
                "build_time_s": build_time,
                "index_memory_mb": memory_bytes / 1e6,
                "n_passages": len(passage_ids),
                "n_queries": len(query_ids),
            })
            print(f"      ef_search={ef_search:>4}  recall={mean_recall:.3f}  "
                  f"qps={qps:8.1f}  p50={rows[-1]['p50_latency_ms']:.2f}ms")
    finally:
        store.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                         help="Run against the tiny dataset produced by "
                              "data_prep.py --smoke-test, using one small scale "
                              "tier instead of config.yaml's full 50K/250K/1M list.")
    parser.add_argument("--systems", nargs="+", default=None,
                         help="Only run these systems, e.g. --systems faiss")
    parser.add_argument("--region", default=None,
                         help="Region this run is physically executing in, e.g. "
                              "ap-south-1. Defaults to run_region in config.yaml.")
    args = parser.parse_args()

    config = load_config()
    region = args.region or config["run_region"]
    print(f"Tagging results as measured in region: {region}")
    data_dir = config["paths"]["data_dir"]
    results_dir = config["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    k = config["dataset"]["top_k"]

    passage_ids_all = np.load(os.path.join(data_dir, "passages_ids.npy"))
    passage_vectors_all = np.load(os.path.join(data_dir, "passages_emb.npy"))
    query_ids = np.load(os.path.join(data_dir, "queries_ids.npy"))
    query_vectors = np.load(os.path.join(data_dir, "queries_emb.npy"))

    if args.smoke_test:
        scales = [{"name": "smoke", "n_passages": min(2000, len(passage_ids_all))}]
    else:
        scales = config["dataset"]["scales"]

    systems = args.systems or [s["name"] for s in config["systems"] if s["enabled"]]

    all_rows = []
    for scale in scales:
        n = min(scale["n_passages"], len(passage_ids_all))
        scale_name = scale["name"]
        print(f"\n=== scale: {scale_name} ({n} passages) ===")
        passage_ids = passage_ids_all[:n]
        passage_vectors = passage_vectors_all[:n]

        ground_truth = load_or_compute_ground_truth(
            scale_name, passage_vectors, query_vectors, k, cache_dir=results_dir
        )

        for system_name in systems:
            print(f"  -- system: {system_name} --")
            t0 = time.time()
            rows = run_one_system_one_scale(
                system_name, config, region, scale_name, passage_ids, passage_vectors,
                query_ids, query_vectors, ground_truth, k,
            )
            all_rows.extend(rows)
            print(f"     done in {time.time() - t0:.1f}s")

    df = pd.DataFrame(all_rows)
    out_path = os.path.join(results_dir, "raw_results.csv")
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        if "region" not in existing.columns:
            # Rows written before region tagging existed. Their region isn't
            # recorded anywhere, so label them honestly rather than guessing.
            existing.insert(0, "region", "unspecified")
        df = pd.concat([existing, df], ignore_index=True)
        # Replace rows from a re-run of the same (region, scale, system) rather
        # than duplicating them, so re-running one system doesn't corrupt the
        # CSV -- and a run in one region never overwrites another region's rows.
        df = df.drop_duplicates(subset=["region", "scale", "system", "ef_search"], keep="last")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
