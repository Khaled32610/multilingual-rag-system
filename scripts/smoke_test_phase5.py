"""
Smoke test for Phase 5 (BM25) — Step 1 + Step 2.

Runs the REAL pipeline end-to-end, no fakes/mocks:

    PDF file -> ChunkingPipeline (Phase 1+2) -> real chunks
             -> BM25Retriever (Phase 5) -> build a real bm25s index
             -> run a few test queries and print the retrieved chunks
                (including the REAL Helsinki-NLP translation model for
                non-English queries, downloaded on first use)
             -> save the index to disk, then reload it from disk and
                confirm a search still works (persistence roundtrip)

This project's real corpus is confirmed mostly-English documents with
mostly-Arabic questions (see Phase 5 / Step 2 design discussion) — the
test PDF used here is a real example of exactly that mismatch (its
extracted text is English despite the "_AR" filename suffix), which is
exactly the case Step 2's query translation exists to handle.

Usage:
    python scripts/smoke_test_phase5.py
    python scripts/smoke_test_phase5.py --pdf path/to/other.pdf --top-k 3
    python scripts/smoke_test_phase5.py --no-translation   # to see Step 1 alone
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rag_system.chunking.pipeline import ChunkingPipeline
from rag_system.retrieval.bm25_index import BM25Index
from rag_system.retrieval.bm25_retriever import BM25Retriever
from rag_system.retrieval.query_translator import QueryTranslator
from rag_system.utils.config import get_settings
from rag_system.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_PDF = "data/raw/01_Enacted_Budget_Law_2024_AR.pdf"

# Arabic queries against the (English-text) test PDF — exactly the
# mismatch Step 2 exists to bridge. "government budget revenue" is kept
# as a control: it's English, so translation must NOT kick in for it.
DEFAULT_QUERIES = [
    "الموازنة العامة للدولة",
    "وزير المالية",
    "government budget revenue",
]


def line(char: str = "-", n: int = 80) -> None:
    print(char * n)


def run(pdf_path: str, top_k: int, queries: list[str], use_translation: bool) -> None:
    settings = get_settings()

    line("=")
    print("PHASE 5 SMOKE TEST — BM25 (Step 1) + Cross-lingual translation (Step 2)")
    print(f"Query translation: {'ENABLED (real model)' if use_translation else 'DISABLED'}")
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

    # --- Step B: real BM25 indexing (Phase 5 / Step 1) ----------------------
    print(f"\n[2/4] Building BM25 index over {len(chunks)} chunks")
    translator = QueryTranslator(model_name=settings.retrieval.translation_model) if use_translation else None
    retriever = BM25Retriever(bm25_index=BM25Index(), query_translator=translator)

    t0 = time.perf_counter()
    added = retriever.index_chunks(chunks)
    t1 = time.perf_counter()

    print(f"    -> {added} chunks indexed in {t1 - t0:.3f}s")
    print(f"    -> index size: {len(retriever)}")

    if added != len(chunks):
        print(f"    WARNING: expected {len(chunks)} indexed, got {added}")

    # --- Step C: real queries against the real index (+ translation) --------
    print(f"\n[3/4] Running {len(queries)} test quer(y/ies), top_k={top_k}")
    print("    (first Arabic query will download the ~300MB translation model if not cached)")
    for q in queries:
        line()
        print(f"QUERY: {q!r}")
        t0 = time.perf_counter()
        results = retriever.retrieve(q, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if not results:
            print("  (no results — none of the query's terms, translated or not, appear in the corpus)")
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
    save_dir = Path(settings.paths.bm25_index_dir) / "_smoke_test"
    retriever.bm25_index.save(save_dir)
    print(f"    -> saved to {save_dir}")

    reloaded_retriever = BM25Retriever(
        bm25_index=BM25Index.load(save_dir), query_translator=translator
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
    print("SMOKE TEST PASSED — Phase 5 (BM25 + translation) works end-to-end with real data.")
    line("=")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5 BM25 smoke test")
    parser.add_argument("--pdf", default=DEFAULT_PDF, help="Path to a real PDF to index")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results per query")
    parser.add_argument(
        "--query", action="append", dest="queries", default=None,
        help="Custom query (repeatable). Defaults to DEFAULT_QUERIES if omitted.",
    )
    parser.add_argument(
        "--no-translation", action="store_true",
        help="Disable query translation, to see Step 1 (BM25 alone) behavior.",
    )
    args = parser.parse_args()

    run(
        pdf_path=args.pdf,
        top_k=args.top_k,
        queries=args.queries or DEFAULT_QUERIES,
        use_translation=not args.no_translation,
    )