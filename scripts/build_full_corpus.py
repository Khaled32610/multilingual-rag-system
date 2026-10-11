"""
Phase 8-B / Step 1: rebuild vector store + BM25 index from ALL documents
in data/raw/ (8 documents), replacing the single-document index from
Phase 6-9.
"""

from pathlib import Path

from rag_system.chunking.pipeline import ChunkingPipeline
from rag_system.retrieval.bm25_index import BM25Index
from rag_system.retrieval.bm25_retriever import BM25Retriever
from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.retrieval.vector_retriever import VectorRetriever
from rag_system.retrieval.vector_store import VectorStore
from rag_system.utils.config import get_settings


def main() -> None:
    settings = get_settings()
    raw_dir = Path(settings.paths.raw_pdf_dir)
    pdf_files = sorted(raw_dir.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF(s) in {raw_dir}:")
    for f in pdf_files:
        print(f"  - {f.name}")

    chunking_pipeline = ChunkingPipeline()
    results = chunking_pipeline.chunk_files([str(f) for f in pdf_files])

    all_chunks = []
    for file_path, chunks in results.items():
        print(f"{Path(file_path).name}: {len(chunks)} chunk(s)")
        all_chunks.extend(chunks)

    print(f"\nTotal chunks across corpus: {len(all_chunks)}")

    # The cleaning or chunking rules may have changed since the last build.
    # Start with empty stores so removed/changed chunks cannot linger.
    hybrid = HybridRetriever(
        vector_retriever=VectorRetriever(
            vector_store=VectorStore(dimension=settings.embedding.dimension)
        ),
        bm25_retriever=BM25Retriever(bm25_index=BM25Index()),
    )
    counts = hybrid.index_chunks(all_chunks)
    print(f"Indexed: {counts}")

    hybrid.save()
    print("Saved vector store + BM25 index to disk.")


if __name__ == "__main__":
    main()