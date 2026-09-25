# llm.py
# RAG Step 6: Configure the LLM and define the prompt template.
#
# The prompt template tells the LLM exactly how to behave:
#   - Use ONLY the provided context to answer.
#   - If the context doesn't contain the answer, say so honestly.
#   - Keep answers concise and grounded in the document.
#
# We use ChatGroq because it is fast, free-tier friendly, and supports
# several capable open-weight models (llama, mixtral, gemma, etc.).

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load GROQ_API_KEY and GROQ_MODEL_NAME from the .env file
load_dotenv()

# ── Prompt template ──────────────────────────────────────────────────────────
# {context}  → the retrieved document chunks (injected by the RAG pipeline)
# {question} → the user's question (injected by the RAG pipeline)

RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions
based strictly on the provided document context.

If the answer is not contained in the context, say:
"I don't have enough information in the document to answer that question."

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def get_llm() -> ChatGroq:
    """
    Build and return a ChatGroq LLM instance using settings from .env.

    Returns:
        ChatGroq: Configured LLM ready for use in the RAG pipeline.

    Raises:
        ValueError: If GROQ_API_KEY is not set in the environment.
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your-groq-api-key-here":
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Please add your key to the .env file. "
            "Get one free at https://console.groq.com"
        )

    # Default model based on available models on your Groq account.
    # Other available options: allam-2-7b, openai/gpt-oss-20b
    model_name = os.getenv("GROQ_MODEL_NAME", "qwen/qwen3.8-27b")

    llm = ChatGroq(
        model=model_name,
        temperature=0,       # 0 = deterministic — good for factual Q&A
        groq_api_key=api_key,
    )

    print(f"[llm] Using Groq model: {model_name}")
    return llm


def get_prompt() -> PromptTemplate:
    """
    Return the RAG prompt template.

    Returns:
        PromptTemplate: Prompt with {context} and {question} placeholders.
    """
    return PROMPT
