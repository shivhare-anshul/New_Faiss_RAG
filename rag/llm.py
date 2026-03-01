"""
Build a context string from retrieved chunks and get an LLM response (OpenAI GPT).

Data flow:
  retrieved: list[str] -> build_context() -> context: str (capped at max_chars)
  (query, context) -> get_llm_response() -> response: str (LLM answer)
"""

import logging
import os

logger = logging.getLogger(__name__)


def get_openai_client():
    """
    Return OpenAI client for chat completions.
    Reads TEAMIFIED_OPENAI_API_KEY or OPENAI_API_KEY from environment.
    Returns: openai.OpenAI instance.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("OpenAI package required: pip install openai")

    api_key = os.environ.get("TEAMIFIED_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set TEAMIFIED_OPENAI_API_KEY (or OPENAI_API_KEY) in the environment or .env file.")
    return OpenAI(api_key=api_key)


def build_context(chunks: list[str], max_chars: int = 4000) -> str:
    """
    Concatenate retrieved chunks into a single context string, up to max_chars.

    Args:
        chunks: list[str]. Retrieved chunks from FAISS. Example: ["chunk1...", "chunk2...", ...]
        max_chars: Maximum total character count. Stops adding chunks when exceeded.

    Returns:
        str. Chunks joined with "\n\n---\n\n". Example:
        "On February 25, 1986, after four days of bloodless EDSA...\n\n---\n\nJosé Rizal was..."
        Length <= max_chars (approximately; may exceed by one chunk).
    """
    parts = []
    total = 0
    for c in chunks:
        if total + len(c) > max_chars:
            break
        parts.append(c.strip())
        total += len(c)
    return "\n\n---\n\n".join(parts)


def get_llm_response(
    query: str,
    context: str,
    model: str | None = None,
) -> str:
    """
    Send context + query to the LLM and return the assistant's answer.

    Args:
        query: User question. Example: "When did the EDSA People Power Revolution happen?"
        context: String built from retrieved chunks (from build_context).
        model: OpenAI chat model. If None, uses env OPENAI_MODEL (default "gpt-4o-mini").

    Returns:
        str. The LLM's answer. Example: "The EDSA People Power Revolution happened on February 25, 1986."
        Empty string if the API returns no content.
    """
    if model is None:
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    logger.info("[llm] Calling LLM model=%s, context_len=%d chars", model, len(context))
    client = get_openai_client()

    system = (
        "You are a helpful assistant that answers questions based only on the provided context "
        "from a document about Philippine History. If the context does not contain enough "
        "information to answer, say so. Keep answers concise and factual."
    )
    user_content = f"""Use the following context to answer the question.

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content or ""
    logger.info("[llm] Response received: %d chars", len(content))
    return content
