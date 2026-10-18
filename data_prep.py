"""
Downloads MS MARCO passages and queries (via the BeIR/msmarco mirror on
Hugging Face, which exposes the classic passage-ranking corpus in a clean
{_id, title, text} format), embeds everything with ONE consistent embedding
model, and saves the result to disk for the benchmark harness to load.

Every scale tier (50K / 250K / 1M passages, per config.yaml) is a PREFIX of
the same embedded array, not a separately-sampled set. This keeps every tier
consistent with every other, and avoids re-embedding the same passages
multiple times.

Usage:
    python data_prep.py                 # full run, sized by config.yaml
    python data_prep.py --smoke-test     # tiny run (500 passages) to verify
                                          # the whole pipeline end-to-end fast

Output (written to config's paths.data_dir):
    passages_ids.npy      -- MS MARCO passage ids, shape (N,)
    passages_emb.npy       -- passage embeddings, shape (N, embedding_dim), float32
    queries_ids.npy        -- MS MARCO query ids, shape (n_queries,)
    queries_emb.npy         -- query embeddings, shape (n_queries, embedding_dim), float32
    queries_text.json       -- query id -> query text, kept for readability in reports
"""

import argparse
import json
import os
import time

import numpy as np
import yaml
from datasets import load_dataset
from fastembed import TextEmbedding
from tqdm import tqdm


def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def embed_in_batches(model, texts, batch_size=256, desc="embedding"):
    vectors = []
    batch = []
    for text in tqdm(texts, desc=desc):
        batch.append(text)
        if len(batch) == batch_size:
            vectors.extend(model.embed(batch))
            batch = []
    if batch:
        vectors.extend(model.embed(batch))
    return np.array(vectors, dtype=np.float32)


def stream_first_n(dataset_iterable, n, text_field="text", id_field="_id"):
    """Returns the first n rows with non-empty text. Empty rows are skipped
    without counting toward n, so a 1M tier really holds 1M passages."""
    ids = []
    texts = []
    n_skipped = 0
    for row in dataset_iterable:
        if len(texts) == n:
            break
        # Some MS MARCO passages have empty text after cleaning; skip those
        # rather than embedding a blank vector that would never be a
        # meaningful nearest neighbor.
        if not row[text_field]:
            n_skipped += 1
            continue
        ids.append(row[id_field])
        texts.append(row[text_field])
    if n_skipped:
        print(f"  (skipped {n_skipped} rows with empty text)")
    if len(texts) < n:
        print(f"  WARNING: dataset ran out after {len(texts)} non-empty rows "
              f"(asked for {n})")
    return ids, texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                         help="Run on a tiny slice (500 passages, 15 queries) "
                              "to verify the pipeline works end-to-end quickly, "
                              "instead of the full config-sized run.")
    args = parser.parse_args()

    config = load_config()
    ds_cfg = config["dataset"]
    data_dir = config["paths"]["data_dir"]
    os.makedirs(data_dir, exist_ok=True)

    n_passages = 500 if args.smoke_test else max(s["n_passages"] for s in ds_cfg["scales"])
    n_queries = 15 if args.smoke_test else ds_cfg["n_queries"]

    print(f"Loading embedding model: {ds_cfg['embedding_model']}")
    model = TextEmbedding(model_name=ds_cfg["embedding_model"])

    print(f"Streaming {n_passages} passages from {ds_cfg['hf_path']} ...")
    corpus = load_dataset(ds_cfg["hf_path"], ds_cfg["hf_corpus_config"],
                           split=ds_cfg["hf_corpus_config"], streaming=True)
    t0 = time.time()
    passage_ids, passage_texts = stream_first_n(corpus, n_passages)
    print(f"  -> got {len(passage_texts)} passages in {time.time() - t0:.1f}s")

    print(f"Streaming {n_queries} queries from {ds_cfg['hf_path']} ...")
    queries = load_dataset(ds_cfg["hf_path"], ds_cfg["hf_queries_config"],
                            split=ds_cfg["hf_queries_config"], streaming=True)
    query_ids, query_texts = stream_first_n(queries, n_queries)
    print(f"  -> got {len(query_texts)} queries")

    print("Embedding passages ...")
    passage_emb = embed_in_batches(model, passage_texts, desc="passages")

    print("Embedding queries ...")
    query_emb = embed_in_batches(model, query_texts, desc="queries")

    np.save(os.path.join(data_dir, "passages_ids.npy"), np.array(passage_ids))
    np.save(os.path.join(data_dir, "passages_emb.npy"), passage_emb)
    np.save(os.path.join(data_dir, "queries_ids.npy"), np.array(query_ids))
    np.save(os.path.join(data_dir, "queries_emb.npy"), query_emb)
    with open(os.path.join(data_dir, "queries_text.json"), "w") as f:
        json.dump(dict(zip(query_ids, query_texts)), f, indent=2)

    print(f"\nSaved to {data_dir}/:")
    print(f"  passages: {passage_emb.shape}  (dim={passage_emb.shape[1]})")
    print(f"  queries:  {query_emb.shape}")
    print("\nDone. Scale tiers are prefixes of this array -- the harness "
          "slices passages_emb[:n] per config.yaml's scales list, so no "
          "further data prep is needed per tier.")


if __name__ == "__main__":
    main()
