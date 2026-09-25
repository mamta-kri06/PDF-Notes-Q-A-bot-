# app.py
# Streamlit UI for the RAG application.
#
# User flow:
#   1. Upload a PDF via the sidebar.
#   2. Click "Process Document" — old PDF + ChromaDB are wiped, new ones created.
#   3. Type a question in the main area and press Enter / click Ask.
#   4. The answer and the retrieved source chunks are displayed.

import os
import shutil

import streamlit as st

from document_loader import load_pdf
from chunking import split_documents
from embeddings import get_embedding_model
from vector_store import (
    create_vector_store,
    load_vector_store,
    delete_vector_store,
    vector_store_exists,
)
from rag_pipeline import build_rag_pipeline, run_query

# ── Constants ────────────────────────────────────────────────────────────────
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG — PDF Q&A",
    page_icon="📄",
    layout="wide",
)

# ── Helper: cache the embedding model so it loads only once per session ──────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embeddings():
    return get_embedding_model()


# ── Helper: cache the RAG chain; rebuild when a new document is processed ────
@st.cache_resource(show_spinner="Building RAG pipeline…")
def load_pipeline(_embedding_model, _rebuild_trigger: int):
    """
    _rebuild_trigger is an integer stored in session_state.
    Incrementing it forces Streamlit to rerun this cached function,
    effectively rebuilding the pipeline after a new document is processed.
    """
    vector_store = load_vector_store(_embedding_model)
    return build_rag_pipeline(vector_store)


# ── Session state initialisation ─────────────────────────────────────────────
if "rebuild_trigger" not in st.session_state:
    st.session_state.rebuild_trigger = 0          # incremented each time a new doc is processed

if "document_ready" not in st.session_state:
    st.session_state.document_ready = vector_store_exists()   # True if a store already exists on disk

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []            # list of {"question": ..., "answer": ..., "sources": ...}

if "current_filename" not in st.session_state:
    st.session_state.current_filename = None


# ── Utility: delete all files in the uploads directory ───────────────────────
def clear_uploads():
    for fname in os.listdir(UPLOAD_DIR):
        fpath = os.path.join(UPLOAD_DIR, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — document upload & processing
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 RAG PDF Q&A")
    st.markdown("Upload a PDF, process it, then ask questions about its content.")
    st.divider()

    # ── File uploader ─────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Only PDF files are supported.",
    )

    # ── Process button ────────────────────────────────────────────────────────
    process_clicked = st.button(
        "⚙️ Process Document",
        disabled=(uploaded_file is None),
        use_container_width=True,
        type="primary",
    )

    if process_clicked and uploaded_file is not None:
        with st.spinner("Processing document…"):
            try:
                # 1. Delete old uploaded files
                clear_uploads()

                # 2. Delete old ChromaDB so we start fresh
                delete_vector_store()

                # 3. Save the new PDF to disk so PyPDFLoader can read it
                save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # 4. Load → chunk → embed → store
                documents = load_pdf(save_path)
                chunks = split_documents(documents)
                embedding_model = load_embeddings()
                create_vector_store(chunks, embedding_model)

                # 5. Update state so the main area rebuilds the pipeline
                st.session_state.rebuild_trigger += 1
                st.session_state.document_ready = True
                st.session_state.current_filename = uploaded_file.name
                st.session_state.chat_history = []   # clear old conversation

                st.success(f"✅ '{uploaded_file.name}' processed successfully!")

            except Exception as e:
                st.error(f"Error processing document: {e}")

    # ── Status indicator ──────────────────────────────────────────────────────
    st.divider()
    if st.session_state.document_ready:
        fname = st.session_state.current_filename or "a previous document"
        st.success(f"**Active document:** {fname}")
    else:
        st.info("No document loaded yet.")

    # ── Clear chat button ─────────────────────────────────────────────────────
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA — question input & answers
# ─────────────────────────────────────────────────────────────────────────────
st.header("💬 Ask a question about your document")

if not st.session_state.document_ready:
    st.info("👈 Upload and process a PDF using the sidebar to get started.")
else:
    # ── Question input ────────────────────────────────────────────────────────
    with st.form(key="question_form", clear_on_submit=True):
        user_question = st.text_input(
            "Your question",
            placeholder="e.g. What is the main topic of this document?",
            label_visibility="collapsed",
        )
        ask_clicked = st.form_submit_button("Ask", use_container_width=True, type="primary")

    if ask_clicked and user_question.strip():
        with st.spinner("Thinking…"):
            try:
                embedding_model = load_embeddings()
                chain = load_pipeline(embedding_model, st.session_state.rebuild_trigger)
                result = run_query(chain, user_question.strip())

                # Prepend to history so newest appears at the top
                st.session_state.chat_history.insert(0, {
                    "question": user_question.strip(),
                    "answer": result["answer"],
                    "sources": result["sources"],
                })

            except ValueError as ve:
                # API key not configured
                st.error(str(ve))
            except Exception as e:
                st.error(f"Error generating answer: {e}")

    # ── Display chat history ──────────────────────────────────────────────────
    for i, entry in enumerate(st.session_state.chat_history):
        st.divider()

        # Question
        st.markdown(f"**🧑 Question:** {entry['question']}")

        # Answer
        st.markdown(f"**🤖 Answer:**\n\n{entry['answer']}")

        # Retrieved source chunks (collapsed by default to keep UI clean)
        with st.expander(f"📚 View retrieved chunks ({len(entry['sources'])} sources)"):
            for j, doc in enumerate(entry["sources"]):
                page = doc.metadata.get("page", "?")
                source = os.path.basename(doc.metadata.get("source", "unknown"))
                st.markdown(f"**Chunk {j + 1}** — *{source}*, page {page + 1}")
                st.text_area(
                    label=f"chunk_{i}_{j}",
                    value=doc.page_content,
                    height=150,
                    disabled=True,
                    label_visibility="collapsed",
                )
