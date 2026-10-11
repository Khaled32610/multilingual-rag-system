"""Local Streamlit chat and PDF management interface."""

from __future__ import annotations

import tempfile
from pathlib import Path

from rag_system.chat.history import ChatHistoryStore
from rag_system.chat.service import RAGChatService
from rag_system.generation.grounded_generator import GroundedGenerator
from rag_system.llm.default import get_default_llm_client
from rag_system.pipeline.rag_pipeline import RAGPipeline
from rag_system.retrieval.hybrid_retriever import HybridRetriever
from rag_system.retrieval.language_utils import detect_language
from rag_system.retrieval.retrieval_pipeline import RetrievalPipeline
from rag_system.utils.document_manager import DocumentManager
from rag_system.utils.exceptions import RAGSystemError
from rag_system.verification.faithfulness_checker import FaithfulnessChecker
from rag_system.verification.pipeline import VerificationPipeline


def _create_services():
    llm = get_default_llm_client()
    hybrid = HybridRetriever()
    retrieval = RetrievalPipeline(hybrid_retriever=hybrid)
    pipeline = RAGPipeline(
        GroundedGenerator(retrieval, llm),
        VerificationPipeline(FaithfulnessChecker(llm)),
    )
    return RAGChatService(pipeline, ChatHistoryStore()), DocumentManager(retriever=hybrid)


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise RuntimeError("Install the UI extra with: pip install -e '.[ui]' ") from exc

    st.set_page_config(page_title="RAG Chat", page_icon="📚", layout="wide")
    st.title("مساعد المستندات | Document Assistant")
    st.caption("إجابات مبنية على ملفات PDF المضافة، مع إظهار المصادر وحالة التحقق.")

    @st.cache_resource
    def services():
        return _create_services()

    chat_service, document_manager = services()
    sessions = chat_service.history.list_sessions(limit=100)
    session_ids = [session.session_id for session in sessions]
    if not session_ids:
        session = chat_service.start_session()
        session_ids = [session.session_id]
        sessions = [session]
        st.session_state.session_id = session.session_id
    if st.session_state.get("session_id") not in session_ids:
        st.session_state.session_id = session_ids[0]

    with st.sidebar:
        st.subheader("المحادثات")
        if st.button("محادثة جديدة", use_container_width=True):
            st.session_state.session_id = chat_service.start_session().session_id
            st.rerun()
        labels = {
            session.session_id: session.title or f"محادثة {session.session_id[:8]}"
            for session in sessions
        }
        selected = st.selectbox(
            "اختر محادثة", session_ids,
            index=session_ids.index(st.session_state.session_id),
            format_func=lambda item: labels.get(item, item[:8]),
        )
        st.session_state.session_id = selected

        st.divider()
        st.subheader("المستندات")
        uploaded = st.file_uploader("أضف ملف PDF", type=["pdf"])
        if uploaded and st.button("فهرسة الملف", use_container_width=True):
            try:
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / Path(uploaded.name).name
                    path.write_bytes(uploaded.getvalue())
                    result = document_manager.add_pdf(path)
                st.success(f"{result['status']}: {result['file_name']}")
                st.rerun()
            except (RAGSystemError, OSError, ValueError) as exc:
                st.error(f"تعذرت فهرسة الملف: {exc}")
        for document in document_manager.list_documents():
            col, remove = st.columns([4, 1])
            col.caption(f"{document['file_name']} · {document['chunk_count']} مقطع")
            if remove.button("حذف", key=f"remove-{document['file_name']}"):
                try:
                    document_manager.remove_document(document["file_name"])
                    st.rerun()
                except (RAGSystemError, OSError, ValueError) as exc:
                    st.error(f"تعذر حذف المستند: {exc}")

    history = chat_service.history.get_messages(st.session_state.session_id)
    for message in history:
        with st.chat_message(message.role):
            st.markdown(message.content)
            if message.role == "assistant" and message.verified is not None:
                st.caption("تم التحقق من الإجابة" if message.verified else "تعذر التحقق من الإجابة؛ عُرض رد آمن")

    query = st.chat_input("اكتب سؤالك عن المستندات")
    if query:
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"), st.spinner("أبحث في الأدلة وأتحقق من الإجابة…"):
                try:
                    response = chat_service.send(st.session_state.session_id, query)
                    st.markdown(response.answer)
                    st.caption("تم التحقق من الإجابة" if response.verified else "تعذر التحقق من الإجابة؛ عُرض رد آمن")
                    if response.draft.chunks:
                        evidence_label = (
                            "المقاطع التالية أدلة للإجابة وليست إجابات بديلة."
                            if detect_language(query) == "ar"
                            else "The following excerpts are evidence, not alternative answers."
                        )
                        st.caption(evidence_label)
                    for index, chunk in enumerate(response.draft.chunks, start=1):
                        metadata = chunk.metadata or {}
                        source_name = metadata.get("source_file", "مستند")
                        with st.expander(f"[مقتطف {index}] {source_name}"):
                            st.write(chunk.text)
                except (RAGSystemError, OSError, ValueError) as exc:
                    # RAGChatService persists user turns before inference; show an explicit retry hint.
                    st.error(f"تعذر إكمال السؤال. يمكنك إرسال السؤال مرة أخرى. التفاصيل: {exc}")


if __name__ == "__main__":
    main()
