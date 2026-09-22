"""
app/services/generation.py
=============================
Turns retrieved chunks + a question into a grounded answer. If
ANTHROPIC_API_KEY is configured, calls Claude with a citation-style prompt.
Otherwise falls back to a clearly-labelled extractive stub, so the API stays
usable end-to-end even before an LLM key is wired in.
"""

import logging
from typing import List

from app.core.config import settings
from app.schemas.query import SourceChunk

logger = logging.getLogger("rag-backend.generation")

PROMPT_TEMPLATE = """You are a document assistant. Answer the question using ONLY the context snippets below.
If the answer is not contained in the context, say "I don't have enough information in the provided documents to answer that."
Cite the snippet number(s) you used, like [1] or [1][3].

Context:
{context_block}

Question: {question}

Answer (with citations):"""


def build_prompt(question: str, chunks: List[SourceChunk]) -> str:
    context_block = "\n\n".join(
        f"[{i + 1}] (source: {c.source}, page {c.page})\n{c.text}"
        for i, c in enumerate(chunks)
    )
    return PROMPT_TEMPLATE.format(context_block=context_block, question=question)


def generate_answer(question: str, chunks: List[SourceChunk]) -> str:
    """
    Returns a grounded answer string. Calls Claude if an API key is
    configured and the call succeeds; otherwise returns the top retrieved
    chunk as a transparent, clearly-labelled fallback.
    """
    if not chunks:
        return "I don't have enough information in the provided documents to answer that."

    if settings.anthropic_api_key:
        try:
            return _generate_with_claude(question, chunks)
        except Exception as e:
            logger.error(f"LLM generation failed, falling back to extractive answer: {e}")

    return _fallback_answer(chunks)


def _generate_with_claude(question: str, chunks: List[SourceChunk]) -> str:
    import anthropic

    prompt = build_prompt(question, chunks)
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.generation_model,
        max_tokens=settings.generation_max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _fallback_answer(chunks: List[SourceChunk]) -> str:
    top = chunks[0]
    return (
        "[No LLM configured — set ANTHROPIC_API_KEY in .env to enable real generation. "
        "Showing top retrieved snippet instead.]\n\n"
        f"{top.text[:500]}"
    )
