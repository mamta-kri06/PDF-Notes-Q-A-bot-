# document_loader.py
# RAG Step 1: Load a PDF file and extract its text as LangChain Document objects.
#
# PyPDFLoader reads the PDF page by page and returns a list of Document objects.
# Each Document has:
#   - page_content : the raw text of that page
#   - metadata     : dict with at least {"source": <file_path>, "page": <page_number>}

from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str) -> list:
    """
    Load a PDF from `file_path` and return a list of LangChain Document objects,
    one per page.

    Args:
        file_path (str): Absolute or relative path to the PDF file.

    Returns:
        list[Document]: Extracted pages as LangChain Documents.
    """
    # PyPDFLoader handles opening, reading, and closing the PDF file
    loader = PyPDFLoader(file_path)

    # .load() returns a list of Document objects (one per page)
    documents = loader.load()

    print(f"[document_loader] Loaded {len(documents)} page(s) from '{file_path}'")
    return documents
