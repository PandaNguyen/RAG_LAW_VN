import time
from typing import Generator, List

import streamlit as st
from langchain_core.documents import Document

from src.generation.rag import RAGPipeline
from src.retrieve.retrieve_rerank import Retriever
from src.vectorstores.qdrant_store import QdrantRetriever


st.set_page_config(
    page_title="Vietnam Legal RAG Chat",
    page_icon="⚖️",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: "Be Vietnam Pro", sans-serif; }
    .stApp {
        background: radial-gradient(1200px 500px at 0% 0%, #fff4db 0%, #ffffff 50%) no-repeat;
    }
    .block-container { padding-top: 1.2rem; }
    .badge {
        display: inline-block;
        border: 1px solid #d6b47a;
        background: #fff7e8;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
    }
    .caption-mono { font-family: "IBM Plex Mono", monospace; color: #66512e; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def build_pipeline(collection_name: str, retrieve_mode: str, top_k: int) -> RAGPipeline:
    vector_store = QdrantRetriever(
        collection_name=collection_name,
        retrieve_mode=retrieve_mode,
        top_k=top_k,
    )
    retriever = Retriever(
        vector_store=vector_store,
        search_mode=retrieve_mode,
        top_k=top_k,
    )
    return RAGPipeline(retriever=retriever)


def stream_text(text: str, delay: float = 0.008) -> Generator[str, None, None]:
    words = text.split(" ")
    for i, word in enumerate(words):
        suffix = " " if i < len(words) - 1 else ""
        yield word + suffix
        time.sleep(delay)


def render_sources(source_documents: List[Document]) -> None:
    if not source_documents:
        st.caption("Không có nguồn tham khảo nào được sử dụng để trả lời câu hỏi này.")
        return

    with st.expander("Nguồn tham khảo", expanded=False):
        for idx, doc in enumerate(source_documents, 1):
            law_id = doc.metadata.get("law_id", "N/A")
            aaid = doc.metadata.get("aaid", "N/A")
            aid = doc.metadata.get("aid", "N/A")
            snippet = (doc.page_content or "").strip()
            snippet = snippet[:350] + ("..." if len(snippet) > 350 else "")
            st.markdown(f"**{idx}. {law_id} - Điều {aaid} - {aid}**")
            if snippet:
                st.write(snippet)


st.markdown("<div class='badge'>Vietnamese Legal Assistant</div>", unsafe_allow_html=True)
st.title("Chatbot Hỏi Đáp Pháp Luật Việt Nam")

with st.sidebar:
    st.header("Cau hinh")
    collection_name = st.text_input("Qdrant collection", value="legal_documents")
    retrieve_mode = st.selectbox("Retrieve mode", options=["hybrid", "dense", "sparse"], index=0)
    top_k = st.slider("Top K", min_value=1, max_value=20, value=5, step=1)
    clear = st.button("Xóa lich sử chat")
    st.divider()
    st.caption("Luu y: can khoi tao collection va set GOOGLE_API_KEY truoc khi chat.")

if clear:
    st.session_state["messages"] = []

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

prompt = st.chat_input("Nhập câu hỏi của bạn về pháp luật Việt Nam...")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Đang truy xuất và tổng hợp câu trả lời..."):
                rag = build_pipeline(collection_name, retrieve_mode, top_k)
                result = rag.query(prompt)

            answer_text = result.get("result", "")
            source_docs = result.get("source_documents", [])
            st.write_stream(stream_text(answer_text))
            render_sources(source_docs)

            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": answer_text,
                    "sources": source_docs,
                }
            )
        except Exception as exc:
            error_text = f"Lỗi khi xử lý câu hỏi: {exc}"
            st.error(error_text)
            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": error_text,
                    "sources": [],
                }
            )
