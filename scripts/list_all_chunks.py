"""Phase 8 / Step 1 helper: dump all chunks with their IDs so we can
manually label ground-truth (query -> relevant chunk_id) pairs."""

from rag_system.retrieval.vector_store import VectorStore
from rag_system.utils.config import get_settings


def main() -> None:
    settings = get_settings()
    store = VectorStore.load(settings.paths.vector_store_dir)

    # _records: int_id -> {"chunk_id": str, "text": str, "metadata": dict}
    # نرتبهم حسب chunk_index في الـ metadata عشان يطلعوا بترتيب المستند
    records = sorted(
        store._records.values(),
        key=lambda r: r["metadata"].get("chunk_index", 0),
    )

    for rec in records:
        page = rec["metadata"].get("page_number", "?")
        idx = rec["metadata"].get("chunk_index", "?")
        print(f"\n--- chunk_id: {rec['chunk_id']}  (page={page}, chunk_index={idx}) ---")
        print(rec["text"][:300])


if __name__ == "__main__":
    main()
