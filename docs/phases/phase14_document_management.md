# Phase 14 — Document upload and incremental indexing

`DocumentManager` accepts local PDF paths and copies new documents into the configured raw-document directory. It checks extension and configured size limits, computes a SHA-256 fingerprint over normalized extracted text, skips documents with the same extracted content even if PDF bytes or filenames differ, and replaces a registered file when the same name has new content.

For additions and replacements it chunks and embeds only the changed PDF, updates FAISS by stable chunk IDs, and rebuilds BM25 from the cached chunks of the full corpus because `bm25s` does not support incremental updates. Deletion removes the document's vectors and rebuilds BM25 from remaining chunks. If an index operation fails, the manager restores the old in-memory index state and persists it again.

On first use, if an existing BM25 or FAISS index predates the document catalog, the manager reconstructs a catalog and per-document chunk cache from index metadata so the first upload does not drop the existing corpus. The catalog and chunk cache are local runtime state and are ignored by Git.

## API

- `DocumentManager.add_pdf(path)` → indexed / replaced / already_indexed result
- `DocumentManager.remove_document(file_name)` → whether a registered document was removed
- `DocumentManager.list_documents()` → names, hashes, sizes, chunk counts

## Limits

BM25 must be rebuilt after each corpus change. Index persistence uses the project's existing local vector/BM25 save APIs; this is not a crash-atomic multi-file transaction. If both an update and its rollback fail, rebuild indexes from the PDFs in the raw directory.

## Files

- `src/rag_system/utils/document_manager.py`
- `tests/unit/test_document_manager.py`
