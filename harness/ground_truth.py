"""
Computes the exact top-k neighbors for every query, against the FULL
passage set at a given scale, using brute-force search. This is what
"correct" means when we score each system's approximate recall -- so it
must be exact, not approximate, and it must be recomputed per scale (the
correct top-10 for 50K passages is not the same as for 1M passages, since
the corpus itself is different).

Results are cached to disk per scale, since brute force is the same cost
every time it's run and every system-under-test needs the same answer key.
Each cache file has a fingerprint sidecar (.meta.json) recording exactly
which vectors it was computed from. The cache is only reused when the
fingerprint matches the current data -- otherwise re-running data_prep.py
(smoke test, different n_queries, different model, ...) would silently
score recall against a stale answer key with no error.
"""

import hashlib
import json
import os

import faiss
import numpy as np


def compute_ground_truth(passage_vectors: np.ndarray, query_vectors: np.ndarray, k: int) -> np.ndarray:
    """Returns an array of shape (n_queries, k): the positional index (into
    passage_vectors) of each query's true top-k nearest neighbors, ranked by
    inner product -- the same metric every system under test uses."""
    dim = passage_vectors.shape[1]
    flat_index = faiss.IndexFlatIP(dim)
    flat_index.add(np.ascontiguousarray(passage_vectors, dtype=np.float32))
    _, positions = flat_index.search(np.ascontiguousarray(query_vectors, dtype=np.float32), k)
    return positions


def data_fingerprint(passage_vectors, query_vectors, k: int) -> dict:
    """Identifies the exact inputs a ground truth was computed from. Hashing
    the raw bytes catches any change to the vectors, not just a change in
    shape (e.g. same count but a different embedding model)."""
    def digest(a):
        return hashlib.blake2b(np.ascontiguousarray(a, dtype=np.float32).tobytes(),
                                digest_size=16).hexdigest()
    return {
        "n_passages": int(passage_vectors.shape[0]),
        "n_queries": int(query_vectors.shape[0]),
        "dim": int(passage_vectors.shape[1]),
        "k": int(k),
        "passages_hash": digest(passage_vectors),
        "queries_hash": digest(query_vectors),
    }


def load_or_compute_ground_truth(scale_name: str, passage_vectors, query_vectors, k: int, cache_dir: str) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"ground_truth_{scale_name}_k{k}.npy")
    meta_path = os.path.join(cache_dir, f"ground_truth_{scale_name}_k{k}.meta.json")
    fingerprint = data_fingerprint(passage_vectors, query_vectors, k)

    if os.path.exists(cache_path):
        cached_fingerprint = None
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                cached_fingerprint = json.load(f)
        if cached_fingerprint == fingerprint:
            return np.load(cache_path)
        # A cache with no sidecar predates fingerprinting, so there's no way
        # to know what it was computed from -- treat it as stale too.
        print(f"    ground truth cache for '{scale_name}' doesn't match current "
              f"data ({fingerprint['n_passages']} passages, "
              f"{fingerprint['n_queries']} queries) -- recomputing")

    ground_truth = compute_ground_truth(passage_vectors, query_vectors, k)
    np.save(cache_path, ground_truth)
    with open(meta_path, "w") as f:
        json.dump(fingerprint, f, indent=2)
    return ground_truth


def recall_at_k(returned_ids, true_positional_neighbors, passage_ids, k: int) -> float:
    """returned_ids: ids returned by the system under test (real MS MARCO ids).
    true_positional_neighbors: positions from compute_ground_truth (positional,
    not real ids) -- converted to real ids via passage_ids before comparing,
    since systems under test return real ids, not array positions."""
    true_ids = {passage_ids[p] for p in true_positional_neighbors[:k]}
    returned = set(returned_ids[:k])
    if not true_ids:
        return 0.0
    return len(returned & true_ids) / len(true_ids)
