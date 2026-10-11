"""Smoke test for Phase 6 — Hybrid Retrieval (RRF fusion)."""

from pathlib import Path

from rag_system.chunking.pipeline import ChunkingPipeline
from rag_system.retrieval.bm25_index import BM25Index
from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.retrieval.vector_store import VectorStore
from rag_system.utils.config import get_settings
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)

QUESTIONS = [
    "الموازنة العامة للدولة",
    "government budget revenue",
    "وزير المالية",
]


def build_and_save_indexes() -> HybridRetriever:
    """يبني فهارس vector + BM25 من الـ PDF الحقيقي ويحفظهم على الديسك."""
    settings = get_settings()
    pdf_path = Path(settings.paths.raw_pdf_dir) / "01_Enacted_Budget_Law_2024_AR.pdf"

    print(f"Ingesting + chunking: {pdf_path}")
    chunking_pipeline = ChunkingPipeline()
    chunks = chunking_pipeline.chunk_file(str(pdf_path))
    print(f"Got {len(chunks)} chunk(s)")

    hybrid = HybridRetriever()
    counts = hybrid.index_chunks(chunks)
    print(f"Indexed: {counts}")

    hybrid.save()
    print("Saved vector store + BM25 index to disk.")
    return hybrid


def main() -> None:
    settings = get_settings()
    has_vector = VectorStore.exists(Path(settings.paths.vector_store_dir))
    has_bm25 = BM25Index.exists(Path(settings.paths.bm25_index_dir))

    if has_vector and has_bm25:
        print("Found existing indexes on disk, loading them.")
        hybrid = HybridRetriever()
    else:
        print("No existing indexes found, building from scratch.")
        hybrid = build_and_save_indexes()

    for q in QUESTIONS:
        print(f"\n{'=' * 70}\nQuery: {q}\n{'=' * 70}")
        results = hybrid.retrieve(q, top_k=5)
        if not results:
            print("  (no results)")
            continue
        for r in results:
            sources = r.metadata.get("rrf_sources", {})
            print(f"  rank={r.rank}  rrf_score={r.score:.5f}  sources={sources}")
            print(f"    text: {r.text[:100]!r}")


if __name__ == "__main__":
    main()
