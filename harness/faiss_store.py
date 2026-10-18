import time

import faiss
import numpy as np

from harness.base import VectorStore


class FaissStore(VectorStore):
    """FAISS runs in-process (no server), using IndexHNSWFlat with inner
    product, matching the HNSW construction used by pgvector and Qdrant so
    the three are index-for-index comparable."""

    name = "faiss"

    def __init__(self):
        self.index = None
        self.id_map = None   # FAISS positions are 0..n-1; map back to real ids
        self.M = None         # FAISS's C++ HNSW object doesn't expose M back

    def build_index(self, ids, vectors, M: int, ef_construction: int) -> float:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        dim = vectors.shape[1]
        self.index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.id_map = np.asarray(ids)
        self.M = M

        start = time.perf_counter()
        self.index.add(vectors)
        return time.perf_counter() - start

    def set_ef_search(self, ef_search: int) -> None:
        self.index.hnsw.efSearch = ef_search

    def search(self, query_vector, k: int):
        q = np.ascontiguousarray(query_vector, dtype=np.float32).reshape(1, -1)
        start = time.perf_counter()
        _, positions = self.index.search(q, k)
        latency = time.perf_counter() - start
        ids = [self.id_map[p] for p in positions[0] if p != -1]
        return ids, latency

    def index_memory_bytes(self) -> int:
        # FAISS doesn't expose a direct memory accessor; approximate from the
        # HNSW graph's own accounting (vectors + graph links), which is the
        # dominant cost and the same approximation used across all systems.
        n = self.index.ntotal
        dim = self.index.d
        vector_bytes = n * dim * 4
        # HNSW stores ~2*M links per node on average across levels, each an
        # int32 neighbor id.
        graph_bytes = n * 2 * self.M * 4
        return vector_bytes + graph_bytes

    def close(self) -> None:
        self.index = None
        self.id_map = None
