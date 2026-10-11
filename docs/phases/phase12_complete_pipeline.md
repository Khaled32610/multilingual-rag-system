# Phase 12 — Complete RAG pipeline

`RAGPipeline` composes the Phase 10 `GroundedGenerator` and Phase 11 `VerificationPipeline`.
It returns the generated answer as the user-facing answer only when evidence, citations, and faithfulness all pass. If verification fails, it returns a language-matched refusal and preserves the generated draft and full verification result for evaluation.

Backend choices remain injectable through the generator and verifier. The pipeline wraps known project errors and unexpected runtime errors as `PipelineError` while preserving the original exception as the cause.

## Files

- `src/rag_system/pipeline/rag_pipeline.py`
- `tests/unit/test_rag_pipeline.py`
