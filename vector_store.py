# vector_store.py
# RAG Step 4: Persist and manage the ChromaDB vector store.
#
# ChromaDB stores document chunks alongside their embedding vectors on disk.
# This lets us do fast similarity searches without re-embedding on every query.
#
# Functions exposed:
#   create_vector_store  – embed chunks and save to disk
#   load_vector_store    – load an existing store from disk
#   delete_vector_store  – wipe the store directory so a fresh one can be created

import os
import shutil

from langchain_community.vectorstores import Chroma

# Default path where ChromaDB files will be written
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")


def create_vector_store(chunks: list, embedding_model) -> Chroma:
    """
    Embed `chunks` and persist them to a ChromaDB vector store on disk.

    Args:
        chunks (list[Document]): Text chunks from the chunking step.
        embedding_model:         HuggingFace embedding model from embeddings.py.

    Returns:
        Chroma: The newly created and persisted vector store.
    """
    print(f"[vector_store] Creating vector store with {len(chunks)} chunk(s)...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_PATH,
    )

    print(f"[vector_store] Vector store saved to '{CHROMA_DB_PATH}'")
    return vector_store


def load_vector_store(embedding_model) -> Chroma:
    """
    Load an existing ChromaDB vector store from disk.

    Args:
        embedding_model: Must be the same model used when the store was created.

    Returns:
        Chroma: The loaded vector store ready for similarity search.
    """
    print(f"[vector_store] Loading vector store from '{CHROMA_DB_PATH}'")

    vector_store = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embedding_model,
    )

    return vector_store


def delete_vector_store() -> None:
    """
    Delete the ChromaDB directory from disk.

    Call this before processing a new document so stale vectors are removed.
    """
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
        print(f"[vector_store] Deleted existing vector store at '{CHROMA_DB_PATH}'")
    else:
        print("[vector_store] No existing vector store found — nothing to delete.")


def vector_store_exists() -> bool:
    """
    Check whether a persisted ChromaDB directory already exists on disk.

    Returns:
        bool: True if the store exists, False otherwise.
    """
    return os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH)
