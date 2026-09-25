# rag_pipeline.py
# RAG Step 7: Connect retrieval + LLM into a single end-to-end pipeline.
#
# The full RAG flow wired here:
#
#   User question
#       │
#       ▼
#   Retriever  ──►  Top-4 relevant chunks from ChromaDB
#       │
#       ▼
#   Prompt template  ──►  Fills {context} and {question}
#       │
#       ▼
#   LLM (ChatOpenAI)  ──►  Generates the answer
#       │
#       ▼
#   Answer string returned to the caller (app.py)
#
# LangChain's RetrievalQA chain handles all of the above automatically.

from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import Chroma

from llm import get_llm, get_prompt


def build_rag_pipeline(vector_store: Chroma) -> RetrievalQA:
    """
    Build a RetrievalQA chain that retrieves context from ChromaDB
    and passes it to the LLM with the RAG prompt.

    Args:
        vector_store (Chroma): A loaded ChromaDB vector store.

    Returns:
        RetrievalQA: A ready-to-use question-answering chain.
    """
    llm = get_llm()
    prompt = get_prompt()

    # Build the retriever (k=4 similarity search, defined in retriever.py logic)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    # RetrievalQA chain:
    #   chain_type="stuff" → concatenates all retrieved chunks into one context block.
    #   return_source_documents=True → also returns the chunks used, so the UI
    #   can display them as sources.
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    print("[rag_pipeline] RAG pipeline ready.")
    return chain


def run_query(chain: RetrievalQA, question: str) -> dict:
    """
    Run a question through the RAG pipeline and return the answer
    along with the source documents used to generate it.

    Args:
        chain (RetrievalQA): The built RAG chain.
        question (str):      The user's question.

    Returns:
        dict with keys:
            "answer"   (str)         – the LLM-generated answer
            "sources"  (list[Document]) – the retrieved chunks used as context
    """
    print(f"[rag_pipeline] Running query: '{question[:80]}'")

    # LangChain returns {"result": "...", "source_documents": [...]}
    response = chain.invoke({"query": question})

    return {
        "answer": response.get("result", "No answer generated."),
        "sources": response.get("source_documents", []),
    }
