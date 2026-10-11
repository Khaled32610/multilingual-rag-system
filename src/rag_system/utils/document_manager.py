"""Local PDF management with incremental vector updates and BM25 rebuilds."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

from rag_system.chunking.models import Chunk, ChunkMetadata
from rag_system.chunking.pipeline import ChunkingPipeline
from rag_system.ingestion.hasher import PDFHasher
from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.utils.config import get_settings
from rag_system.utils.exceptions import DocumentManagementError, IndexingError
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)


class DocumentManager:
    """Add, replace, list, and remove PDFs from the local RAG corpus.

    Vector embeddings are added/deleted per document. Since bm25s does not
    support incremental mutation, its index is rebuilt from the cached chunks
    of all remaining documents whenever the corpus changes.
    """

    def __init__(
        self,
        chunker: ChunkingPipeline | None = None,
        retriever: HybridRetriever | None = None,
        raw_dir: str | Path | None = None,
        catalog_path: str | Path | None = None,
    ) -> None:
        settings = get_settings()
        self.raw_dir = Path(raw_dir or settings.paths.raw_pdf_dir)
        self.catalog_path = Path(
            catalog_path or (Path(settings.paths.data_dir) / "document_catalog.json")
        )
        self.chunk_cache_dir = self.catalog_path.parent / "document_chunks"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_cache_dir.mkdir(parents=True, exist_ok=True)
        self.chunker = chunker or ChunkingPipeline()
        self.retriever = retriever or HybridRetriever()
        self.hasher = PDFHasher()
        self._catalog = self._load_catalog()
        if not self._catalog:
            self._bootstrap_from_existing_index()

    def _load_catalog(self) -> dict[str, dict[str, Any]]:
        if not self.catalog_path.exists():
            return {}
        try:
            value = json.loads(self.catalog_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise TypeError("catalog root must be a JSON object")
            return value
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            raise DocumentManagementError(
                "Could not read document catalog", details={"error": str(exc)}
            ) from exc

    def _bootstrap_from_existing_index(self) -> None:
        """Create the document catalog/cache from a pre-Phase-14 BM25 or FAISS index."""
        bm25_index = self.retriever.bm25_retriever.bm25_index
        records = getattr(bm25_index, "_records", getattr(bm25_index, "records", []))
        if not records:
            records = list(
                getattr(self.retriever.vector_retriever.vector_store, "_records", {}).values()
            )
        grouped: dict[tuple[str, str], list[Chunk]] = {}
        for record in records:
            metadata = record.get("metadata", {})
            filename = metadata.get("source_file")
            digest = metadata.get("document_hash")
            if not filename or not digest:
                continue
            try:
                chunk = Chunk(
                    text=record["text"],
                    chunk_id=record["chunk_id"],
                    metadata=ChunkMetadata(
                        source_file=filename,
                        document_hash=digest,
                        page_number=int(metadata.get("page_number", 1)),
                        chunk_index=int(metadata.get("chunk_index", 0)),
                        total_chunks_in_doc=int(metadata.get("total_chunks_in_doc", 0)),
                        char_count=int(metadata.get("char_count", len(record["text"]))),
                        created_at=metadata.get(
                            "created_at", "1970-01-01T00:00:00+00:00"
                        ),
                    ),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise IndexingError(
                    "Could not reconstruct chunks from the existing retrieval index",
                    details={"error": str(exc)},
                ) from exc
            grouped.setdefault((filename, digest), []).append(chunk)

        if not grouped:
            return
        catalog: dict[str, dict[str, Any]] = {}
        for (filename, digest), chunks in grouped.items():
            self._atomic_json(
                self.chunk_cache_dir / f"{digest}.json",
                [self._serialize_chunk(chunk) for chunk in chunks],
            )
            document_path = self.raw_dir / filename
            catalog[filename] = {
                "document_hash": digest,
                "content_hash": self._content_hash(chunks),
                "size_bytes": document_path.stat().st_size if document_path.is_file() else None,
                "chunk_count": len(chunks),
            }
        self._atomic_json(self.catalog_path, catalog)
        self._catalog = catalog

    @staticmethod
    def _content_hash(chunks: list[Chunk]) -> str:
        """Fingerprint extracted document text, independent of the PDF filename/encoding."""
        canonical = "\n".join(
            " ".join(unicodedata.normalize("NFKC", chunk.text).split())
            for chunk in sorted(chunks, key=lambda item: item.metadata.chunk_index)
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _serialize_chunk(chunk: Chunk) -> dict[str, Any]:
        return chunk.to_dict()

    @staticmethod
    def _deserialize_chunk(data: dict[str, Any]) -> Chunk:
        metadata = ChunkMetadata(**data["metadata"])
        return Chunk(text=data["text"], metadata=metadata, chunk_id=data["chunk_id"])

    def _chunks_for(self, filename: str) -> list[Chunk]:
        cache_path = self.chunk_cache_dir / f"{self._catalog[filename]['document_hash']}.json"
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return [self._deserialize_chunk(item) for item in payload]
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise IndexingError(
                f"Cached chunks unavailable for {filename}", details={"error": str(exc)}
            ) from exc

    def _all_chunks(self, exclude: set[str] | None = None) -> list[Chunk]:
        exclude = exclude or set()
        return [
            chunk
            for filename in sorted(self._catalog)
            if filename not in exclude
            for chunk in self._chunks_for(filename)
        ]

    @staticmethod
    def _atomic_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, path)
        except Exception:
            Path(temp_name).unlink(missing_ok=True)
            raise

    def list_documents(self) -> list[dict[str, Any]]:
        """Return registered document names, hashes, sizes, and chunk counts."""
        return [
            {"file_name": name, **metadata}
            for name, metadata in sorted(self._catalog.items())
        ]

    def add_pdf(self, source_path: str | Path) -> dict[str, Any]:
        """Copy and index a PDF. Same-name files are replaced; identical content is skipped."""
        source = Path(source_path)
        if not source.is_file():
            raise DocumentManagementError(f"PDF file does not exist: {source}")
        if source.suffix.lower() not in get_settings().pdf.allowed_extensions:
            raise DocumentManagementError(f"Unsupported document type: {source.suffix}")

        max_bytes = get_settings().pdf.max_file_size_mb * 1024 * 1024
        if source.stat().st_size > max_bytes:
            raise DocumentManagementError(
                f"PDF exceeds configured size limit ({get_settings().pdf.max_file_size_mb} MB)"
            )
        file_digest = self.hasher.compute_hash(source)
        filename = source.name
        previous_entry = self._catalog.get(filename)
        previous_chunks = self._chunks_for(filename) if previous_entry else []
        managed_path = self.raw_dir / filename
        temp_copy = self.raw_dir / f".{Path(filename).stem}.{file_digest[:12]}.upload.pdf"
        try:
            shutil.copy2(source, temp_copy)
            chunks = self.chunker.chunk_file(str(temp_copy))
            if not chunks:
                raise DocumentManagementError(f"No text chunks extracted from {filename}")
            digest = self._content_hash(chunks)
            existing_name = next(
                (name for name, entry in self._catalog.items() if entry.get("content_hash", entry["document_hash"]) == digest),
                None,
            )
            if existing_name is not None:
                temp_copy.unlink(missing_ok=True)
                return {"status": "already_indexed", "file_name": existing_name, **self._catalog[existing_name]}
            # Identity follows normalized extracted text; filename is display metadata.
            for chunk in chunks:
                chunk.metadata.source_file = filename
                chunk.metadata.document_hash = digest
                chunk.chunk_id = chunk._generate_id()
            backup_path = None
            if managed_path.exists():
                backup_path = self.raw_dir / f".{filename}.{file_digest[:12]}.previous"
                os.replace(managed_path, backup_path)
            os.replace(temp_copy, managed_path)

            try:
                self._commit_corpus_change(filename, chunks, previous_chunks)
            except Exception:
                managed_path.unlink(missing_ok=True)
                if backup_path and backup_path.exists():
                    os.replace(backup_path, managed_path)
                raise
            if backup_path:
                backup_path.unlink(missing_ok=True)
        except (DocumentManagementError, IndexingError):
            temp_copy.unlink(missing_ok=True)
            raise
        except Exception as exc:
            temp_copy.unlink(missing_ok=True)
            raise IndexingError(
                f"Failed to add or index {filename}", details={"error": str(exc)}
            ) from exc

        return {"status": "indexed" if not previous_entry else "replaced", "file_name": filename, **self._catalog[filename]}

    def _commit_corpus_change(
        self, filename: str, new_chunks: list[Chunk] | None, old_chunks: list[Chunk]
    ) -> None:
        previous_catalog = self._catalog.copy()
        remaining = self._all_chunks(exclude={filename})
        next_catalog = previous_catalog.copy()
        if new_chunks:
            remaining.extend(new_chunks)
            digest = new_chunks[0].metadata.document_hash
            cache_path = self.chunk_cache_dir / f"{digest}.json"
            self._atomic_json(cache_path, [self._serialize_chunk(chunk) for chunk in new_chunks])
            next_catalog[filename] = {
                "document_hash": digest,
                "content_hash": digest,
                "size_bytes": (self.raw_dir / filename).stat().st_size,
                "chunk_count": len(new_chunks),
            }
        else:
            next_catalog.pop(filename, None)

        old_ids = [chunk.chunk_id for chunk in old_chunks]
        new_ids = {chunk.chunk_id for chunk in new_chunks or []}
        try:
            if old_ids:
                self.retriever.vector_retriever.vector_store.delete_by_chunk_ids(old_ids)
            if new_chunks:
                self.retriever.vector_retriever.index_chunks(new_chunks)
            if remaining:
                self.retriever.bm25_retriever.index_chunks(remaining)
            else:
                self.retriever.bm25_retriever.bm25_index.clear()
            self.retriever.save()
            self._atomic_json(self.catalog_path, next_catalog)
            self._catalog = next_catalog
            if old_chunks and old_chunks[0].metadata.document_hash != (new_chunks[0].metadata.document_hash if new_chunks else None):
                old_cache = self.chunk_cache_dir / f"{old_chunks[0].metadata.document_hash}.json"
                old_cache.unlink(missing_ok=True)
        except Exception as exc:
            try:
                if new_ids:
                    self.retriever.vector_retriever.vector_store.delete_by_chunk_ids(list(new_ids))
                if old_chunks:
                    self.retriever.vector_retriever.index_chunks(old_chunks)
                if previous_catalog:
                    self.retriever.bm25_retriever.index_chunks(self._all_chunks_from_catalog(previous_catalog))
                else:
                    self.retriever.bm25_retriever.bm25_index.clear()
                self.retriever.save()
            except Exception as rollback_exc:
                raise IndexingError(
                    "Corpus update failed and rollback also failed; rebuild the indexes from data/raw",
                    details={"update_error": str(exc), "rollback_error": str(rollback_exc)},
                ) from rollback_exc
            raise IndexingError("Corpus update failed; previous index restored", details={"error": str(exc)}) from exc

    def _all_chunks_from_catalog(self, catalog: dict[str, dict[str, Any]]) -> list[Chunk]:
        current = self._catalog
        self._catalog = catalog
        try:
            return self._all_chunks()
        finally:
            self._catalog = current

    def remove_document(self, file_name: str) -> bool:
        """Remove a managed PDF from disk and both retrieval indexes."""
        if Path(file_name).name != file_name:
            raise DocumentManagementError("file_name must be a plain file name")
        if file_name not in self._catalog:
            return False
        path = self.raw_dir / file_name
        old_chunks = self._chunks_for(file_name)
        backup_path = path.with_name(f".{path.name}.delete-backup")
        try:
            if path.exists():
                os.replace(path, backup_path)
            self._commit_corpus_change(file_name, None, old_chunks)
            backup_path.unlink(missing_ok=True)
            return True
        except Exception:
            if backup_path.exists():
                os.replace(backup_path, path)
            raise
