import time

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from harness.base import VectorStore


class QdrantStore(VectorStore):
    """Qdrant has no persistent 'set ef_search' setting like pgvector's GUC --
    ef_search (hnsw_ef) is passed per search request instead, so
    set_ef_search() just stores it for search() to use on the next call."""

    name = "qdrant"

    def __init__(self, host: str, port: int, collection: str):
        self.client = QdrantClient(host=host, port=port)
        self.collection = collection
        self._ef_search = 64  # overwritten by set_ef_search() before use

    def build_index(self, ids, vectors, M: int, ef_construction: int) -> float:
        dim = vectors.shape[1]
        start = time.perf_counter()
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.DOT),
            hnsw_config=qm.HnswConfigDiff(m=M, ef_construct=ef_construction),
        )
        batch_size = 1000
        for start_idx in range(0, len(ids), batch_size):
            batch_ids = ids[start_idx:start_idx + batch_size]
            batch_vecs = vectors[start_idx:start_idx + batch_size]
            self.client.upsert(
                collection_name=self.collection,
                points=[
                    qm.PointStruct(id=int(i), vector=v.tolist())
                    for i, v in zip(batch_ids, batch_vecs)
                ],
                wait=True,
            )
        # Block until Qdrant's segment optimizer finishes building the HNSW
        # graph -- without this, build time would be understated and early
        # queries would run against a partially-built index.
        while True:
            info = self.client.get_collection(self.collection)
            if info.status == qm.CollectionStatus.GREEN:
                break
            time.sleep(0.2)
        return time.perf_counter() - start

    def set_ef_search(self, ef_search: int) -> None:
        self._ef_search = ef_search

    def search(self, query_vector, k: int):
        start = time.perf_counter()
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector.tolist(),
            limit=k,
            search_params=qm.SearchParams(hnsw_ef=self._ef_search),
        )
        latency = time.perf_counter() - start
        ids = [r.id for r in results]
        return ids, latency

    def index_memory_bytes(self) -> int:
        info = self.client.get_collection(self.collection)
        # Newer Qdrant server versions report this directly; fall back to
        # the same analytical estimate used for FAISS if the field isn't
        # present, so the three systems stay comparable even across
        # Qdrant versions.
        reported = getattr(info, "ram_data_size", None) or getattr(info, "disk_data_size", None)
        if reported:
            return int(reported)
        n = info.points_count or 0
        dim = info.config.params.vectors.size
        m = info.config.hnsw_config.m
        return n * dim * 4 + n * 2 * m * 4

    def close(self) -> None:
        self.client.delete_collection(self.collection)
