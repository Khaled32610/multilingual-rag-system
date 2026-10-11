"""
FAISS-backed vector store (Phase 4 / Step 16).

Design decisions (see docs/benchmarks/embedding_benchmark_results.md for
the Phase 3 context these follow from):

- **IndexFlatIP, not an approximate index (HNSW/IVF).** Embeddings are
  L2-normalized before being added (see
  `SentenceTransformerEmbeddingModel`, which calls `encode(...,
  normalize_embeddings=True)`), so inner product is mathematically
  equivalent to cosine similarity. A flat index does an exhaustive
  brute-force scan, which guarantees exact (100% recall) nearest
  neighbours relative to the embedding space itself. This project's
  stated priority is accuracy over latency, and everything runs
  locally, so trading query speed for an approximate index is not
  worth it at the corpus sizes this project targets.
- **Stable int64 ids via IndexIDMap2.** FAISS only stores raw vectors;
  it doesn't know about chunk text or metadata. Each chunk's
  deterministic `chunk_id` (a hex string, from `Chunk._generate_id`) is
  mapped to an internal int64 id assigned on insertion, which
  `IndexIDMap2` lets us address directly for both search results and
  deletion. Text and metadata live outside FAISS, in a parallel dict
  keyed by that same int64 id, persisted as JSON next to the index.
  This is what will let Phase 14 (incremental indexing) delete or
  replace individual chunks without rebuilding the whole index.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import faiss
import numpy as np

from rag_system.chunking.models import Chunk
from rag_system.retrieval.models import RetrievedChunk
from rag_system.utils.exceptions import VectorStoreError
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)

_INDEX_FILE_NAME = "faiss.index"
_METADATA_FILE_NAME = "metadata.json"


class VectorStore:
    """Exact-search (flat, brute-force) vector store backed by FAISS."""

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise VectorStoreError(f"Invalid embedding dimension: {dimension}")

        self.dimension = dimension
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))

        # int64 id -> {"chunk_id": str, "text": str, "metadata": dict}
        self._records: dict[int, dict] = {}
        # chunk_id (deterministic sha string, from Chunk.chunk_id) -> int64 id
        self._chunk_id_to_int: dict[str, int] = {}
        self._next_id: int = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return self._index.ntotal

    # --- Add -----------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        """
        Add chunks and their pre-computed embedding vectors to the store.

        Chunks whose `chunk_id` is already present are skipped (not
        duplicated) rather than overwritten — call `delete_by_chunk_ids`
        first if the intent is to replace an existing chunk's vector.

        Returns the number of vectors actually added.
        """
        if len(chunks) == 0:
            return 0
        if len(chunks) != vectors.shape[0]:
            raise VectorStoreError(
                "Number of chunks does not match number of vectors",
                details={"num_chunks": len(chunks), "num_vectors": int(vectors.shape[0])},
            )
        if vectors.shape[1] != self.dimension:
            raise VectorStoreError(
                "Vector dimension does not match store dimension",
                details={"expected": self.dimension, "got": int(vectors.shape[1])},
            )

        with self._lock:
            new_chunks, new_rows, new_ids = [], [], []
            skipped = 0

            for chunk, vector in zip(chunks, vectors):
                if chunk.chunk_id in self._chunk_id_to_int:
                    skipped += 1
                    continue
                int_id = self._next_id
                self._next_id += 1
                self._chunk_id_to_int[chunk.chunk_id] = int_id
                self._records[int_id] = {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata.to_dict(),
                }
                new_chunks.append(chunk)
                new_rows.append(vector)
                new_ids.append(int_id)

            if new_rows:
                matrix = np.asarray(new_rows, dtype="float32")
                ids_array = np.asarray(new_ids, dtype="int64")
                try:
                    self._index.add_with_ids(matrix, ids_array)
                except Exception as e:
                    # Roll back bookkeeping so the store stays consistent
                    # with what's actually in the FAISS index.
                    for int_id in new_ids:
                        rec = self._records.pop(int_id, None)
                        if rec:
                            self._chunk_id_to_int.pop(rec["chunk_id"], None)
                    raise VectorStoreError(
                        "Failed to add vectors to FAISS index", details={"error": str(e)}
                    ) from e

        if skipped:
            logger.warning(f"Skipped {skipped} chunk(s) already present in the vector store")
        logger.info(
            f"Added {len(new_rows)} chunk(s) to vector store (total={self._index.ntotal})"
        )
        return len(new_rows)

    # --- Search ----------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[RetrievedChunk]:
        """Return the top-k chunks by cosine similarity to `query_vector`."""
        if top_k <= 0:
            raise VectorStoreError(f"top_k must be positive, got {top_k}")

        if self._index.ntotal == 0:
            logger.warning("Vector store search called on an empty index")
            return []

        query = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        if query.shape[1] != self.dimension:
            raise VectorStoreError(
                "Query vector dimension does not match store dimension",
                details={"expected": self.dimension, "got": int(query.shape[1])},
            )

        k = min(top_k, self._index.ntotal)
        try:
            scores, ids = self._index.search(query, k)
        except Exception as e:
            raise VectorStoreError("FAISS search failed", details={"error": str(e)}) from e

        results: list[RetrievedChunk] = []
        for rank, (int_id, score) in enumerate(zip(ids[0], scores[0]), start=1):
            if int_id == -1:
                continue  # FAISS pads with -1 when fewer than k matches exist
            record = self._records.get(int(int_id))
            if record is None:
                logger.warning(f"Vector store id {int_id} missing from metadata records")
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=record["chunk_id"],
                    text=record["text"],
                    metadata=record["metadata"],
                    score=float(score),
                    rank=rank,
                    source="vector",
                )
            )
        return results

    # --- Delete (used by Phase 14 incremental indexing) -------------------

    def delete_by_chunk_ids(self, chunk_ids: list[str]) -> int:
        """Remove chunks by their stable chunk_id. Returns number removed."""
        with self._lock:
            int_ids = [
                self._chunk_id_to_int[cid] for cid in chunk_ids if cid in self._chunk_id_to_int
            ]
            if not int_ids:
                return 0

            selector = np.asarray(int_ids, dtype="int64")
            try:
                removed = self._index.remove_ids(selector)
            except Exception as e:
                raise VectorStoreError(
                    "Failed to remove vectors from FAISS index", details={"error": str(e)}
                ) from e

            for int_id in int_ids:
                rec = self._records.pop(int_id, None)
                if rec:
                    self._chunk_id_to_int.pop(rec["chunk_id"], None)

        logger.info(f"Removed {removed} chunk(s) from vector store")
        return int(removed)

    def has_chunk(self, chunk_id: str) -> bool:
        return chunk_id in self._chunk_id_to_int

    # --- Persistence -------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        try:
            faiss.write_index(self._index, str(directory / _INDEX_FILE_NAME))
        except Exception as e:
            raise VectorStoreError(
                f"Failed to write FAISS index to {directory}", details={"error": str(e)}
            ) from e

        payload = {
            "dimension": self.dimension,
            "next_id": self._next_id,
            "records": {str(k): v for k, v in self._records.items()},
        }
        with (directory / _METADATA_FILE_NAME).open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved vector store ({self._index.ntotal} vectors) to {directory}")

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        directory = Path(directory)
        index_path = directory / _INDEX_FILE_NAME
        metadata_path = directory / _METADATA_FILE_NAME

        if not index_path.exists() or not metadata_path.exists():
            raise VectorStoreError(f"No vector store found at {directory}")

        with metadata_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        store = cls(dimension=payload["dimension"])
        try:
            store._index = faiss.read_index(str(index_path))
        except Exception as e:
            raise VectorStoreError(
                f"Failed to read FAISS index from {index_path}", details={"error": str(e)}
            ) from e

        store._records = {int(k): v for k, v in payload["records"].items()}
        store._chunk_id_to_int = {
            rec["chunk_id"]: int(k) for k, rec in payload["records"].items()
        }
        store._next_id = payload["next_id"]

        logger.info(f"Loaded vector store ({store._index.ntotal} vectors) from {directory}")
        return store

    @classmethod
    def exists(cls, directory: str | Path) -> bool:
        directory = Path(directory)
        return (directory / _INDEX_FILE_NAME).exists() and (
            directory / _METADATA_FILE_NAME
        ).exists()
