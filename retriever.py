# retriever.py
# RAG Step 5: Retrieve the most relevant chunks for a user query.
#
# How it works:
#   1. The user's question is embedded into a vector using the same model.
#   2. ChromaDB computes cosine similarity between the query vector and every
#      stored chunk vector.
#   3. The top-k most similar chunks are returned as context for the LLM.

from langchain_community.vectorstores import Chroma

# Number of chunks to retrieve per query
K = 4


def get_retriever(vector_store: Chroma):
    """
    Build a LangChain retriever from the vector store.

    The retriever wraps similarity_search so it can be used inside
    LangChain chains and pipelines.

    Args:
        vector_store (Chroma): A loaded or newly created ChromaDB store.

    Returns:
        VectorStoreRetriever: LangChain retriever configured with k=4.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",   # Plain cosine / L2 similarity search
        search_kwargs={"k": K},     # Return the 4 most relevant chunks
    )

    print(f"[retriever] Retriever ready (k={K})")
    return retriever


def retrieve_chunks(vector_store: Chroma, query: str) -> list:
    """
    Directly return the top-k most relevant Document chunks for `query`.

    This is used by the Streamlit UI to display source chunks alongside
    the generated answer.

    Args:
        vector_store (Chroma): The vector store to search.
        query (str):           The user's question.

    Returns:
        list[Document]: Up to K Document objects with page_content and metadata.
    """
    results = vector_store.similarity_search(query, k=K)
    print(f"[retriever] Retrieved {len(results)} chunk(s) for query: '{query[:60]}...'")
    return results
