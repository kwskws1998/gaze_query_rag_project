from __future__ import annotations

import numpy as np

from gaze_query_rag.retrieval.index import build_in_memory_index, search_index
from gaze_query_rag.schemas import Chunk, QAExample, RetrievalResult


def _subset_embeddings(
    chunks: list[Chunk], embeddings: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    missing = [chunk.chunk_id for chunk in chunks if chunk.chunk_id not in embeddings]
    if missing:
        raise KeyError(f"Missing embeddings for chunks: {missing[:10]}")
    return {chunk.chunk_id: embeddings[chunk.chunk_id] for chunk in chunks}


def retrieve_text_only(
    example: QAExample,
    chunks: list[Chunk],
    text_embeddings: dict[str, np.ndarray],
    query: np.ndarray,
    top_k: int,
) -> RetrievalResult:
    index = build_in_memory_index(_subset_embeddings(chunks, text_embeddings))
    ranked = search_index(index, query, top_k)
    return RetrievalResult(
        example_id=example.example_id,
        reader_id=None,
        condition="text",
        ranked_chunks=ranked,
        metadata={"paragraph_id": example.paragraph_id},
    )


def retrieve_gaze_view(
    example: QAExample,
    reader_id: str,
    chunks: list[Chunk],
    gaze_embeddings: dict[str, np.ndarray],
    query: np.ndarray,
    top_k: int,
    condition: str,
) -> RetrievalResult:
    if condition not in {"actual_gaze", "mean_gaze", "shuffled_gaze"}:
        raise ValueError(f"Unsupported gaze retrieval condition: {condition}")
    index = build_in_memory_index(_subset_embeddings(chunks, gaze_embeddings))
    ranked = search_index(index, query, top_k)
    return RetrievalResult(
        example_id=example.example_id,
        reader_id=reader_id,
        condition=condition,
        ranked_chunks=ranked,
        metadata={"paragraph_id": example.paragraph_id},
    )
