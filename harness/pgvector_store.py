import time

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from harness.base import VectorStore


class PgVectorStore(VectorStore):
    """Uses pgvector's native HNSW index (pgvector >= 0.5.0), with inner
    product as the distance metric -- the same metric FAISS and Qdrant use
    here, so all three rank neighbors identically for the same vectors.

    build_index() times the FULL path from raw vectors to a queryable index
    (table creation, bulk load, CREATE INDEX), not just the CREATE INDEX
    statement alone. That's the fair comparison point against FAISS, where
    index.add() does loading and indexing in one call."""

    name = "pgvector"

    def __init__(self, dsn: str, table: str):
        self.conn = psycopg2.connect(dsn)
        self.conn.autocommit = True
        self.table = table
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        register_vector(self.conn)

    def build_index(self, ids, vectors, M: int, ef_construction: int) -> float:
        dim = vectors.shape[1]
        start = time.perf_counter()
        with self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.table};")
            cur.execute(
                f"CREATE TABLE {self.table} (id BIGINT PRIMARY KEY, embedding VECTOR({dim}));"
            )
            rows = [(int(i), v.tolist()) for i, v in zip(ids, vectors)]
            psycopg2.extras.execute_values(
                cur,
                f"INSERT INTO {self.table} (id, embedding) VALUES %s",
                rows,
                template="(%s, %s)",
                page_size=1000,
            )
            cur.execute(
                f"CREATE INDEX ON {self.table} USING hnsw (embedding vector_ip_ops) "
                f"WITH (m = {M}, ef_construction = {ef_construction});"
            )
        return time.perf_counter() - start

    def set_ef_search(self, ef_search: int) -> None:
        with self.conn.cursor() as cur:
            cur.execute(f"SET hnsw.ef_search = {ef_search};")

    def search(self, query_vector, k: int):
        with self.conn.cursor() as cur:
            start = time.perf_counter()
            cur.execute(
                f"SELECT id FROM {self.table} ORDER BY embedding <#> %s LIMIT %s;",
                (query_vector, k),
            )
            ids = [row[0] for row in cur.fetchall()]
            latency = time.perf_counter() - start
        return ids, latency

    def index_memory_bytes(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT pg_relation_size(indexrelid) FROM pg_index "
                "JOIN pg_class ON pg_class.oid = pg_index.indrelid "
                "WHERE pg_class.relname = %s;",
                (self.table,),
            )
            row = cur.fetchone()
            return row[0] if row else 0

    def close(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.table};")
        self.conn.close()
