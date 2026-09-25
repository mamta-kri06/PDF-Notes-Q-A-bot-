# RAG Application — Interview Questions & Answers

> Covers concepts, design decisions, and "why this, not that" reasoning
> based on the stack: Python · Streamlit · LangChain · ChromaDB · HuggingFace · Groq

---

## SECTION 1 — RAG Fundamentals

---

**Q1. What is RAG? Why do we need it?**

RAG (Retrieval-Augmented Generation) combines a retrieval system with a generative LLM.
Instead of asking the LLM to answer purely from its training data, you first fetch
relevant chunks from a document and pass them as context to the LLM.

**Why we need it:**
- LLMs have a knowledge cutoff — they don't know about your private or recent documents.
- Fine-tuning is expensive and slow; RAG is plug-and-play.
- RAG reduces hallucinations because the LLM is anchored to retrieved evidence.
- You can update the knowledge base (add/remove docs) without retraining the model.

---

**Q2. What are the main steps in a RAG pipeline?**

```
Document → Load → Chunk → Embed → Store → Retrieve → Prompt → LLM → Answer
```

1. **Load** — Read the raw document (PDF, text, etc.)
2. **Chunk** — Split into smaller pieces so each fits in a vector search
3. **Embed** — Convert each chunk to a numeric vector using an embedding model
4. **Store** — Save vectors + text in a vector database (ChromaDB)
5. **Retrieve** — On a user query, embed the query and find the most similar chunks
6. **Prompt** — Inject retrieved chunks into a prompt template
7. **Generate** — LLM reads the context and produces an answer

---

**Q3. What is the difference between RAG and fine-tuning?**

| | RAG | Fine-tuning |
|---|---|---|
| Knowledge update | Add/remove docs instantly | Retrain the model |
| Cost | Low (only inference) | High (GPU training) |
| Hallucination risk | Lower (grounded in docs) | Higher (baked into weights) |
| Use case | Private/changing data | Changing model behaviour/style |
| Transparency | Sources visible | Black box |

**Interview tip:** RAG is preferred when the data changes frequently or is private.
Fine-tuning is better when you want the model to behave differently (tone, format, domain language).

---

**Q4. What is a vector embedding? Why is it useful?**

An embedding is a fixed-size list of numbers (a vector) that represents the semantic
meaning of a piece of text. Texts with similar meaning have vectors that are close
together in vector space (measured by cosine similarity or dot product).

Example: "What is machine learning?" and "Define ML" will have very similar vectors
even though the words are different.

**Why useful in RAG:** Instead of keyword search (exact word match), we do semantic
search — finding chunks that *mean* the same thing as the query.

---

**Q5. What is cosine similarity? How does it relate to retrieval?**

Cosine similarity measures the angle between two vectors:

```
similarity = (A · B) / (|A| × |B|)
```

- Result ranges from -1 (opposite) to 1 (identical direction).
- In RAG, the query vector is compared to every chunk vector.
- The chunks with the highest cosine similarity are returned as context.

We use `normalize_embeddings=True` in our app so the vectors are unit length,
making cosine similarity equivalent to a simple dot product (faster).

---

## SECTION 2 — Document Loading & Chunking

---

**Q6. Why do we use PyPDFLoader? Why not just open() the PDF?**

PDFs are binary files — they are not plain text. Opening with `open()` gives raw bytes.
PyPDFLoader uses the `pypdf` library internally to:
- Parse the binary PDF structure
- Extract text page by page
- Attach metadata (source file path, page number) to each page

**Why PyPDFLoader specifically:**
- It is the simplest LangChain-native PDF loader
- Works without any external binary dependencies (no Poppler, no Java)
- Returns LangChain `Document` objects ready for the next pipeline step

---

**Q7. Why do we chunk documents? Why not send the whole PDF to the LLM?**

Three reasons:

1. **Context window limit** — LLMs have a maximum token limit (e.g., 8K, 32K tokens).
   A 100-page PDF easily exceeds this.

2. **Cost** — Sending thousands of tokens on every query is expensive.

3. **Quality** — Smaller, focused chunks give the LLM cleaner context.
   Passing an entire book and asking about one paragraph degrades answer quality.

---

**Q8. Why RecursiveCharacterTextSplitter? Why not CharacterTextSplitter?**

`CharacterTextSplitter` splits only on one separator (e.g., `\n`). If a paragraph
has no newlines it won't split at all, producing huge chunks.

`RecursiveCharacterTextSplitter` tries a list of separators in order:
`["\n\n", "\n", " ", ""]`

It keeps trying smaller separators until the chunk is within the size limit.
This means chunks stay **semantically coherent** (paragraph → sentence → word level).

**Why chunk_size=1000 and chunk_overlap=200:**
- 1000 characters ≈ ~250 tokens — fits comfortably in most LLM prompts alongside 4 chunks.
- 200-character overlap ensures a sentence cut at a boundary is still present
  in the adjacent chunk, preventing lost context.

---

**Q9. What happens if chunk_overlap is 0?**

Sentences that happen to fall exactly on a chunk boundary get split in two.
The first half is in chunk N, the second half in chunk N+1.
If only chunk N is retrieved, the answer is incomplete or misleading.
Overlap acts as a safety margin for boundary sentences.

---

## SECTION 3 — Embeddings

---

**Q10. Why sentence-transformers/all-MiniLM-L6-v2? Why not OpenAI embeddings?**

| | all-MiniLM-L6-v2 | OpenAI text-embedding-3-small |
|---|---|---|
| Cost | Free, runs locally | Paid per token |
| Privacy | Data stays on your machine | Sent to OpenAI's servers |
| Speed | Fast on CPU | Requires network round-trip |
| Size | ~80 MB | No download needed |
| Quality | Good for most tasks | Slightly better on benchmarks |

For a learning project or internal tool, MiniLM is the right call.
For production with high accuracy requirements, OpenAI embeddings are worth the cost.

---

**Q11. What does normalize_embeddings=True do?**

It scales each vector to unit length (magnitude = 1).
This makes cosine similarity equivalent to dot product, which is faster to compute.
It also ensures that chunk length doesn't accidentally influence similarity —
a long chunk won't have a larger vector magnitude than a short one.

---

**Q12. Can you use a different embedding model for query vs. documents?**

No — you must use the **same model** for both. The vector space is model-specific.
Using Model A for documents and Model B for queries would put vectors in
incompatible spaces, making similarity scores meaningless.

---

**Q13. What is the dimension of all-MiniLM-L6-v2 embeddings?**

384 dimensions. This is smaller than OpenAI's 1536-dimension embeddings,
which makes it faster to store and search, with minimal quality loss for most tasks.

---

## SECTION 4 — Vector Store (ChromaDB)

---

**Q14. What is ChromaDB? Why ChromaDB and not FAISS or Pinecone?**

ChromaDB is an open-source, embedded vector database that persists data to disk.

| | ChromaDB | FAISS | Pinecone |
|---|---|---|---|
| Type | Embedded (local) | In-memory library | Cloud SaaS |
| Persistence | Yes (SQLite + files) | No (manual save) | Yes (cloud) |
| Setup | pip install, zero config | pip install | API key + account |
| Cost | Free | Free | Paid above free tier |
| Scale | Small-medium | Large (RAM-bound) | Enterprise |
| Best for | Prototypes, local apps | Research, large datasets | Production SaaS |

ChromaDB is the easiest to get started with — no server, no config file,
just point it at a directory and it works.

---

**Q15. How does ChromaDB store data?**

ChromaDB uses a combination of:
- **SQLite** — stores document text and metadata
- **HNSW index** (via hnswlib) — stores and indexes the embedding vectors for fast ANN search

The `persist_directory` you specify becomes a folder containing these files.

---

**Q16. What is ANN search? How is it different from exact search?**

- **Exact search (KNN)** — compare the query vector to every single stored vector. O(n). Accurate but slow at scale.
- **ANN (Approximate Nearest Neighbour)** — uses an index (like HNSW) to find
  *approximately* the nearest neighbours much faster. O(log n). Slight accuracy trade-off.

HNSW (Hierarchical Navigable Small World) builds a multi-layer graph of vectors.
At query time it navigates the graph rather than scanning everything.

For our use case (hundreds to thousands of chunks), even exact search is fast enough.
But ChromaDB uses HNSW by default because it scales well.

---

**Q17. Why do we delete the old ChromaDB before processing a new document?**

ChromaDB persists data across sessions. If we just added new chunks on top of old ones:
- The old document's chunks would still be returned in search results
- Answers would mix content from two different documents
- Users would get incorrect, confusing answers

Deleting and recreating ensures the vector store only contains chunks
from the currently uploaded document.

---

## SECTION 5 — Retrieval

---

**Q18. Why k=4? What happens if k is too small or too large?**

k=4 is a common default that balances context quality vs. prompt size.

- **Too small (k=1 or 2):** Risk of missing the relevant chunk entirely.
  The query embedding might not perfectly match the chunk containing the answer.

- **Too large (k=10+):** The prompt becomes very long, increasing cost and
  potentially diluting the relevant context with irrelevant chunks.
  LLMs can get "distracted" by noise.

k=4 gives the LLM 4 different perspectives/sections from the document,
which is usually enough for a well-answerable question.

---

**Q19. What is the difference between similarity search and MMR retrieval?**

- **Similarity search** — returns the k most similar chunks. Problem: they might all
  be near-duplicates of each other (same section of the document repeated).

- **MMR (Maximal Marginal Relevance)** — balances relevance AND diversity.
  It picks each next chunk to be relevant to the query but different from
  already selected chunks.

For this app we use plain similarity search because our chunks don't overlap much.
MMR is useful for longer documents with repeated concepts.

---

**Q20. What if the answer is not in the document?**

The prompt template explicitly instructs the LLM:

> "If the answer is not contained in the context, say: I don't have enough
> information in the document to answer that question."

Without this instruction the LLM might hallucinate an answer using its training data.
This is called **prompt guardrailing**.

---

## SECTION 6 — LLM & Prompt

---

**Q21. Why Groq? Why not OpenAI or a local LLM?**

| | Groq | OpenAI | Local LLM (Ollama) |
|---|---|---|---|
| Speed | Very fast (LPU hardware) | Moderate | Slow on CPU |
| Cost | Free tier available | Pay per token | Free |
  | Privacy | Data leaves device | Data leaves device | Fully private |
| Setup | API key only | API key only | Download model (~4GB+) |
| Model quality | Llama 3.3 70B (excellent) | GPT-4o (excellent) | Depends on model |

Groq runs LLaMA and Mixtral models on custom LPU (Language Processing Unit) chips
that are much faster than GPU inference — you get GPT-4-class answers at near-instant speed.
It's the best choice when you want fast + free + high quality.

---

**Q22. What is temperature=0 and why do we use it here?**

Temperature controls randomness in the LLM's output:
- **temperature=0** — fully deterministic, always picks the most probable next token
- **temperature=1** — normal randomness
- **temperature>1** — very random/creative

For document Q&A, we want consistent, factual answers — not creative variations.
temperature=0 ensures that asking the same question twice gives the same answer.

---

**Q23. What is a PromptTemplate? Why not just use an f-string?**

A `PromptTemplate` is a LangChain abstraction that:
- Declares named input variables (`{context}`, `{question}`)
- Validates that all variables are provided before sending to the LLM
- Is composable — can be chained with other LangChain components
- Can be serialized/loaded from YAML for sharing across projects

An f-string would work but bypasses LangChain's validation and chaining system.

---

**Q24. What is the "stuff" chain type in RetrievalQA?**

LangChain offers several ways to handle multiple retrieved chunks:

| Chain type | What it does | When to use |
|---|---|---|
| **stuff** | Concatenates all chunks into one prompt | Small chunks, short docs |
| map_reduce | Summarises each chunk separately, then combines | Very long documents |
| refine | Iteratively refines the answer chunk by chunk | When order matters |
| map_rerank | Scores each chunk's answer, picks the best | High accuracy needed |

We use **stuff** because with k=4 and chunk_size=1000, the total context is
~4000 characters — well within any modern LLM's context window.

---

## SECTION 7 — Streamlit & Architecture

---

**Q25. Why Streamlit? Why not Flask or FastAPI?**

| | Streamlit | Flask/FastAPI |
|---|---|---|
| Purpose | Data/ML UI apps | General web APIs |
| Code needed | ~50 lines for a full UI | 200+ lines + HTML/CSS/JS |
| State management | Built-in session_state | Manual (session, cookies) |
| Deployment | `streamlit run app.py` | Configure WSGI/ASGI server |
| Best for | Prototypes, ML demos | Production APIs |

For a RAG demo or internal tool, Streamlit gets you a polished UI in minutes.
Flask/FastAPI would be better if you needed a REST API consumed by another frontend.

---

**Q26. What is st.cache_resource and why do we use it?**

`@st.cache_resource` caches objects that are expensive to create and safe to share
across all users and sessions (e.g., ML models, database connections).

Without it:
- The embedding model (~80 MB) would be reloaded on every user interaction
- The RAG chain would be rebuilt on every question

With it:
- The model loads once and stays in memory
- Subsequent calls return the cached object instantly

We use `_rebuild_trigger` as a cache key so the pipeline is rebuilt only when
a new document is processed, not on every question.

---

**Q27. What is session_state in Streamlit?**

Streamlit re-runs the entire script top-to-bottom on every user interaction.
`st.session_state` is a dictionary that persists values across those reruns.

We use it to store:
- `chat_history` — list of past Q&A pairs
- `document_ready` — whether a document has been processed
- `rebuild_trigger` — integer incremented to force pipeline rebuild
- `current_filename` — name of the currently loaded document

---

**Q28. What is the overall architecture of this application?**

```
User (browser)
    │
    ▼
Streamlit (app.py) ──────────────────────────────────────────────
    │                                                            │
    │  Upload PDF          Ask Question                         │
    ▼                          ▼                                │
document_loader.py         rag_pipeline.py                      │
    │                          │                                │
    ▼                          ├── vector_store.py (load)       │
chunking.py                    │       │                        │
    │                          │       ▼                        │
    ▼                          │   retriever (k=4)              │
embeddings.py                  │       │                        │
    │                          │       ▼                        │
    ▼                          │   llm.py (Groq)                │
vector_store.py (create)       │       │                        │
                               │       ▼                        │
                               └── Answer + Sources ────────────┘
```

---

## SECTION 8 — Advanced / Tricky Questions

---

**Q29. What are the limitations of this RAG implementation?**

1. **Single document only** — Processing a new doc deletes the old one. No multi-doc support.
2. **No conversation memory** — Each question is independent; the LLM doesn't remember previous Q&A.
3. **No reranking** — Retrieved chunks are used as-is. A reranker (e.g., cross-encoder) could improve precision.
4. **Fixed chunk size** — Different document types (tables, code, lists) might need different chunking strategies.
5. **No hybrid search** — Only semantic search, no keyword (BM25) fallback for exact term matching.

---

**Q30. How would you add conversation memory to this RAG app?**

Replace `RetrievalQA` with `ConversationalRetrievalChain` and pass a
`ConversationBufferMemory` object:

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

chain = ConversationalRetrievalChain.from_llm(
    llm=get_llm(),
    retriever=retriever,
    memory=memory,
)
```

The chain automatically includes previous Q&A turns in the prompt context.

---

**Q31. How would you support multiple documents?**

Instead of deleting the vector store on each upload, assign each document a
unique `collection_name` in ChromaDB:

```python
Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=CHROMA_DB_PATH,
    collection_name=filename_without_ext,
)
```

Then let the user select which document(s) to query, and merge retrievers
or use a router chain.

---

**Q32. What is hallucination in LLMs and how does RAG reduce it?**

Hallucination is when an LLM generates a confident-sounding but factually wrong answer.
It happens because LLMs predict plausible tokens, not verified facts.

RAG reduces hallucination by:
1. Providing explicit context — the LLM has real text to quote from
2. Prompt instructions — "Answer only from the context provided"
3. Source transparency — showing users which chunks were used, so they can verify

RAG does **not eliminate** hallucination — the LLM can still misread or
misinterpret the context. But it reduces the rate significantly.

---

**Q33. What is the difference between an embedding model and an LLM?**

| | Embedding Model | LLM |
|---|---|---|
| Input | Text | Text (prompt) |
| Output | Fixed-size vector (numbers) | Text (tokens) |
| Purpose | Represent meaning numerically | Generate/complete text |
| Size | Small (80MB–500MB) | Large (1GB–100GB+) |
| Examples | all-MiniLM-L6-v2, text-embedding-3 | LLaMA, GPT-4, Mixtral |
| Used for | Similarity search | Question answering |

In our pipeline both are used:
- Embedding model: converts chunks + query to vectors for ChromaDB search
- LLM (Groq/LLaMA): reads retrieved chunks and generates the answer

---

**Q34. What would you change to make this production-ready?**

1. **Authentication** — Add user login (Streamlit-Authenticator or OAuth)
2. **Multi-user isolation** — Separate vector store collections per user
3. **Async processing** — Use background tasks for document ingestion (FastAPI + Celery)
4. **Reranking** — Add a cross-encoder reranker after retrieval for better precision
5. **Evaluation** — Use RAGAs framework to measure faithfulness and answer relevance
6. **Observability** — Add LangSmith or Langfuse for tracing LLM calls
7. **Error handling** — Retry logic for API calls, graceful degradation
8. **Containerize** — Dockerfile + docker-compose for consistent deployment

---

*Good luck with your interview!*
