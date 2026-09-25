# embeddings.py
# RAG Step 3: Create a HuggingFace embedding model.
#
# Embeddings convert text into dense numeric vectors so that semantically
# similar pieces of text end up close together in vector space.
# This is what makes similarity search possible.
#
# We use "sentence-transformers/all-MiniLM-L6-v2" because:
#   - It is lightweight (~80 MB) and fast on CPU.
#   - It produces 384-dimensional vectors with good semantic quality.
#   - It runs locally — no API key required for embeddings.

import os
from langchain_huggingface import HuggingFaceEmbeddings

# Prefer local model (downloaded once to the project folder) to avoid
# HuggingFace Hub cache / symlink issues on Windows.
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")
HF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Load and return the HuggingFace embedding model.

    Tries the locally downloaded copy first; falls back to the HuggingFace
    Hub name if the project-level copy isn't present.

    Returns:
        HuggingFaceEmbeddings: A LangChain-compatible embedding model.
    """
    if os.path.isdir(LOCAL_MODEL_PATH) and os.path.exists(
        os.path.join(LOCAL_MODEL_PATH, "config_sentence_transformers.json")
    ):
        model_source = LOCAL_MODEL_PATH
    else:
        model_source = HF_MODEL_NAME

    print(f"[embeddings] Loading embedding model from: {model_source}")

    embedding_model = HuggingFaceEmbeddings(
        model_name=model_source,
        model_kwargs={"device": "cpu"},   # Use "cuda" if a GPU is available
        encode_kwargs={"normalize_embeddings": True},  # Normalise for cosine similarity
    )

    return embedding_model
