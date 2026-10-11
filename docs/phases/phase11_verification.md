# Phase 11 — Evidence and citation verification

## Verification flow

1. `EvidenceGate` rejects an empty answer and rejects a substantive answer when retrieval returned no chunks. A complete, recognized refusal is allowed without a citation.
2. `CitationValidator` accepts Arabic `[مقتطف N]` and English `[Excerpt N]` markers, requires citations for substantive answers, and rejects out-of-range chunk numbers.
3. `FaithfulnessChecker` passes only the cited chunks to an injected local LLM judge. It requires a JSON verdict and fails closed on malformed output or an LLM error. Original citation numbers are preserved in the judge's evidence block.
4. `VerificationPipeline` runs these checks in order and returns a structured `VerificationResult`; unsupported claims fail verification.

## Limits

The LLM faithfulness verdict is a probabilistic check, not a proof of truth. It only evaluates support in the cited excerpts and does not catch errors already present in a source document. The judge is injected so a separate model can be selected later; using the generation model as its own judge may share its biases.

## Files

- `src/rag_system/verification/evidence_gate.py`
- `src/rag_system/verification/citation_validator.py`
- `src/rag_system/verification/faithfulness_checker.py`
- `src/rag_system/verification/pipeline.py`
- `tests/unit/test_verification.py`
