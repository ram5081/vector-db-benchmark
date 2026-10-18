"""
Every system under test (pgvector, Qdrant, FAISS) implements this same
interface. The runner never knows which concrete system it's talking to --
that's what makes the comparison fair: every system is driven through
exactly the same sequence of calls, with exactly the same HNSW parameters.
"""

from abc import ABC, abstractmethod
import time


class VectorStore(ABC):
    name: str

    @abstractmethod
    def build_index(self, ids, vectors, M: int, ef_construction: int) -> float:
        """Build the index from scratch. Returns build time in seconds.
        `ids` and `vectors` are aligned: ids[i] is the id of vectors[i]."""
        raise NotImplementedError

    @abstractmethod
    def set_ef_search(self, ef_search: int) -> None:
        """Change the search-time recall/latency knob without rebuilding."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query_vector, k: int):
        """Run one query. Returns (list_of_ids, latency_seconds)."""
        raise NotImplementedError

    @abstractmethod
    def index_memory_bytes(self) -> int:
        """Approximate resident memory used by the index, in bytes."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release resources / drop the index so the next system starts clean."""
        raise NotImplementedError

    def search_many(self, query_vectors, k: int):
        """Convenience wrapper: run search() over a batch, one at a time --
        this measures single-query latency (p50/p95/p99), which is what a
        live API actually experiences, rather than a batch throughput number
        that would hide per-query tail latency."""
        all_ids = []
        latencies = []
        for q in query_vectors:
            ids, latency = self.search(q, k)
            all_ids.append(ids)
            latencies.append(latency)
        return all_ids, latencies


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start
