# RAG PDF Q&A — Streamlit App

A beginner-friendly **Retrieval-Augmented Generation (RAG)** application that lets you upload a PDF and ask questions about it. Built with Streamlit, LangChain, ChromaDB, and HuggingFace embeddings.

---

## How it works

```
PDF upload
    │
    ▼
PyPDFLoader          ← document_loader.py
    │  (pages)
    ▼
RecursiveCharacterTextSplitter  ← chunking.py
    │  (chunks)
    ▼
HuggingFace Embeddings          ← embeddings.py
    │  (vectors)
    ▼
ChromaDB (persisted on disk)    ← vector_store.py
    │
    ▼  (user asks a question)
Similarity Search (k=4)         ← retriever.py
    │  (top-4 relevant chunks)
    ▼
Prompt Template + ChatOpenAI    ← llm.py
    │  (answer)
    ▼
Streamlit UI                    ← app.py
```

---

## Prerequisites

- Python 3.10 or higher
- A free [Groq API key](https://console.groq.com)

---

## Installation

### 1. Clone or download this project

```bash
cd rag-app
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Open `.env` and replace the placeholder with your real Groq key:

```env
GROQ_API_KEY=gsk_...your-key-here...
GROQ_MODEL_NAME=llama-3.3-70b-versatile
```

Get a free key at [https://console.groq.com](https://console.groq.com).

---

## Running the app

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## Usage

1. **Upload a PDF** using the sidebar file uploader.
2. Click **⚙️ Process Document** — the app embeds the document and stores it in ChromaDB.
   - Processing a new document automatically deletes the old PDF and old ChromaDB.
3. **Type a question** in the main area and click **Ask**.
4. The app displays the **answer** and the **retrieved source chunks** used to generate it.

---

## Project structure

```
rag-app/
├── app.py               # Streamlit UI
├── document_loader.py   # Load PDF with PyPDFLoader
├── chunking.py          # Split pages into chunks
├── embeddings.py        # HuggingFace embedding model
├── vector_store.py      # ChromaDB create / load / delete
├── retriever.py         # Similarity search (k=4)
├── llm.py               # ChatOpenAI + prompt template
├── rag_pipeline.py      # RetrievalQA chain
├── requirements.txt
├── .env                 # API keys (never commit this!)
├── data/uploads/        # Uploaded PDFs saved here
└── chroma_db/           # ChromaDB vector store (auto-created)
```

---

## Configuration

| Setting | File | Default |
|---|---|---|
| LLM model | `.env` → `GROQ_MODEL_NAME` | `llama-3.3-70b-versatile` |
| Chunk size | `chunking.py` | `1000` characters |
| Chunk overlap | `chunking.py` | `200` characters |
| Embedding model | `embeddings.py` | `sentence-transformers/all-MiniLM-L6-v2` |
| Retrieved chunks | `retriever.py` | `k=4` |
| ChromaDB path | `vector_store.py` | `./chroma_db/` |

---

## Notes

- The HuggingFace embedding model is downloaded automatically on first run (~80 MB) and cached locally.
- Embeddings run on CPU by default. Change `"device": "cpu"` to `"cuda"` in `embeddings.py` if you have a GPU.
- Never commit your `.env` file. Add it to `.gitignore`.
