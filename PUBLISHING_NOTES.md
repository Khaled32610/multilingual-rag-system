# Repository Publishing Notes

This is a clean source-only copy of the RAG project prepared for an initial private GitHub upload.

- Runtime PDFs, uploaded documents, persisted chat, FAISS/BM25 indices, logs, benchmark-generated reports and caches were intentionally excluded.
- The original `README.md` describes local native and Docker setup.
- Use the Streamlit UI to upload your own PDFs and rebuild the indices.
- The models must be downloaded on first use; subsequent use can be local/offline.
- Do not commit keys, local user data, or private documents.
- This archive has not been validated by running its full test suite or Docker build.
- Before changing repo visibility from Private to Public, inspect git history and test for secrets.
