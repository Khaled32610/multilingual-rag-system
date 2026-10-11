"""
Smoke test for Phase 4 (Vector Search) — Step 16 + Step 17.

Runs the REAL pipeline end-to-end, no fakes/mocks:

    PDF file -> ChunkingPipeline (Phase 1+2) -> real chunks
             -> VectorRetriever (Phase 4) using the real default
                embedding model selected in Phase 3
                (intfloat/multilingual-e5-base)
             -> index chunks into a fresh FAISS vector store
             -> run a few test queries and print the retrieved chunks
             -> save the index to disk, then reload it from disk and
                confirm a search still works (persistence roundtrip)

Usage:
    python scripts/smoke_test_phase4.py
    python scripts/smoke_test_phase4.py --pdf path/to/other.pdf --top-k 3

This is NOT a pytest test — it's meant to be read while it runs, so you
can eyeball whether the retrieved chunks actually look relevant to each
query. It downloads/loads the real embedding model, so the first run
will be slow (model download) and it needs your GPU/CPU to encode text.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rag_system.chunking.pipeline import ChunkingPipeline
from rag_system.retrieval.vector_retriever import VectorRetriever
from rag_system.retrieval.vector_store import VectorStore
from rag_system.utils.config import get_settings
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_PDF = "data/raw/01_Enacted_Budget_Law_2024_AR.pdf"

# A few test queries in both Arabic and English — edit these to match
# content you know is actually in your test PDF, so you can judge
# whether the top results are genuinely relevant.
DEFAULT_QUERIES = [
    "الموازنة العامة للدولة",
    "الإعفاءات الضريبية",
    "government budget revenue",
]


def line(char: str = "-", n: int = 80) -> None:
    print(char * n)


def run(pdf_path: str, top_k: int, queries: list[str]) -> None:
    settings = get_settings()

    line("=")
    print("PHASE 4 SMOKE TEST — Vector Search (Step 16 + 17)")
    line("=")

    # --- Step A: real chunking (Phase 1 + Phase 2) ------------------------
    print(f"\n[1/4] Chunking real PDF: {pdf_path}")
    t0 = time.perf_counter()
    chunking_pipeline = ChunkingPipeline()
    chunks = chunking_pipeline.chunk_file(pdf_path)
    t1 = time.perf_counter()

    if not chunks:
        print("FAILED: no chunks produced. Check the PDF path and Phase 1/2 output.")
        sys.exit(1)

    print(f"    -> {len(chunks)} chunks produced in {t1 - t0:.2f}s")
    print(f"    -> example chunk[0]: {chunks[0]!r}")

    # --- Step B: real embedding + indexing (Phase 3 model + Phase 4 store) -
    print(f"\n[2/4] Embedding + indexing {len(chunks)} chunks "
          f"with model: {settings.embedding.default_model}")
    t0 = time.perf_counter()
    # Fresh in-memory store for this run, so repeated runs are reproducible
    # and don't just keep re-adding the same PDF to a store on disk.
    retriever = VectorRetriever(vector_store=VectorStore(dimension=settings.embedding.dimension))
    added = retriever.index_chunks(chunks)
    t1 = time.perf_counter()

    print(f"    -> {added} vectors added in {t1 - t0:.2f}s "
          f"({added / (t1 - t0):.1f} chunks/sec)")
    print(f"    -> vector store size: {len(retriever)}")

    if added != len(chunks):
        print(f"    WARNING: expected {len(chunks)} added, got {added} "
              f"(duplicates skipped? check chunk_ids)")

    # --- Step C: real queries against the real store ------------------------
    print(f"\n[3/4] Running {len(queries)} test quer(y/ies), top_k={top_k}")
    for q in queries:
        line()
        print(f"QUERY: {q!r}")
        t0 = time.perf_counter()
        results = retriever.retrieve(q, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if not results:
            print("  (no results)")
            continue

        for r in results:
            preview = r.text[:150].replace("\n", " ")
            print(
                f"  rank={r.rank}  score={r.score:.4f}  "
                f"page={r.metadata.get('page_number')}  "
                f"chunk_id={r.chunk_id[:12]}..."
            )
            print(f"      text: {preview}...")
        print(f"  (query latency: {elapsed_ms:.1f} ms)")
    line()

    # --- Step D: persistence roundtrip --------------------------------------
    print("\n[4/4] Persistence roundtrip: save -> reload -> search again")
    save_dir = Path(settings.paths.vector_store_dir) / "_smoke_test"
    retriever.vector_store.save(save_dir)
    print(f"    -> saved to {save_dir}")

    reloaded_store = VectorStore.load(save_dir)
    reloaded_retriever = VectorRetriever(
        embedding_model=retriever.embedding_model, vector_store=reloaded_store
    )
    check_query = queries[0]
    before = retriever.retrieve(check_query, top_k=1)
    after = reloaded_retriever.retrieve(check_query, top_k=1)

    if before and after and before[0].chunk_id == after[0].chunk_id:
        print(f"    -> OK: top result for {check_query!r} matches before/after reload "
              f"(chunk_id={after[0].chunk_id[:12]}...)")
    else:
        print("    FAILED: top result differs after reload — investigate save/load logic")
        sys.exit(1)

    line("=")
    print("SMOKE TEST PASSED — Phase 4 (Vector Search) works end-to-end with real data.")
    line("=")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 vector search smoke test")
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="Path to a real PDF to index")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results per query")
    parser.add_argument(
        "--query", action="append", dest="queries", default=None,
        help="Custom query (repeatable). Defaults to DEFAULT_QUERIES if omitted.",
    )
    args = parser.parse_args()

    run(pdf_path=args.pdf, top_k=args.top_k, queries=args.queries or DEFAULT_QUERIES)
