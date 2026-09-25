# chunking.py
# RAG Step 2: Split large documents into smaller, overlapping chunks.
#
# Why chunk?
#   LLMs have a limited context window. Splitting the text into smaller pieces
#   lets us retrieve only the most relevant sections instead of sending the
#   entire document to the LLM.
#
# RecursiveCharacterTextSplitter tries to split on paragraphs → sentences →
# words in order, so chunks stay semantically coherent.

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents: list) -> list:
    """
    Split a list of LangChain Documents into smaller chunks.

    Args:
        documents (list[Document]): Full-page documents from document_loader.

    Returns:
        list[Document]: Smaller text chunks, each still carrying the original metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,    # Maximum characters per chunk
        chunk_overlap=200,  # Characters shared between consecutive chunks
                            # (overlap helps preserve context at boundaries)
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    print(f"[chunking] Split {len(documents)} page(s) into {len(chunks)} chunk(s)")
    return chunks
