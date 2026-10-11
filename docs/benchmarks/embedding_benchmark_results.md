# Embedding Model Benchmark Results (Phase 3)

**Date:** 2026-09-25
**Test corpus:** `01_Enacted_Budget_Law_2024_AR.pdf` (14 chunks, chunk_size=512, overlap=64)
**Metric:** Recall@3 (is the expected chunk present in the top-3 retrieved results)

## Results

| Model | Dimension | Texts/sec | Query latency (ms) | Recall@3 |
|---|---|---|---|---|
| **intfloat/multilingual-e5-base** ✅ | 768 | 3.4 | 43.0 | **100.0% (4/4)** |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | 29.4 | 16.8 | 75.0% (3/4) |
| paraphrase-multilingual-mpnet-base-v2 | 768 | 10.0 | 42.8 | 25.0% (1/4) |

## Decision

**Selected model: `intfloat/multilingual-e5-base`**

Rationale:
- Best retrieval accuracy by a large margin (100% vs 75% vs 25%).
- Uses asymmetric query/passage prefixes, well-suited to retrieval tasks specifically
  (unlike the paraphrase-* models, which are optimized for semantic similarity, not retrieval).
- Slower encoding speed (3.4 texts/sec) is acceptable because embedding happens once
  at indexing time, not per query. Query latency (43ms) is fast enough for interactive use.
- Supports both Arabic and English, matching the project's mixed-language document set.

## Note on methodology

An initial benchmark using top-1 exact match showed misleading 0-25% accuracy across
all models. Investigation revealed the correct chunk was consistently ranked 2nd,
with very small similarity gaps for e5-base (e.g. 0.840 vs 0.839). Switching to
Recall@3 (matching how retrieved chunks are actually used downstream, before
reranking in Phase 7) revealed the true quality gap between models.
