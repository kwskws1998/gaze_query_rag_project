from __future__ import annotations

import numpy as np

from gaze_query_rag.modeling.gaze_query_attention import gaze_query_attention, pool_chunk_embedding
from gaze_query_rag.schemas import Chunk


SKIP_HARD_CONDITION = "actual_skip_hard"


def word_skip_to_token_query_weights(
    word_skip: np.ndarray, word_to_token_indices: list[list[int]], num_tokens: int
) -> np.ndarray:
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive.")
    word_values = np.asarray(word_skip, dtype=np.float64)
    if len(word_values) != len(word_to_token_indices):
        raise ValueError("word_skip and word_to_token_indices must have the same length.")

    weights = np.zeros(num_tokens, dtype=np.float64)
    counts = np.zeros(num_tokens, dtype=np.float64)
    for value, token_indices in zip(word_values, word_to_token_indices):
        if not token_indices:
            continue
        skip = 1.0 if np.nan_to_num(value, nan=0.0) >= 0.5 else 0.0
        inspect = 1.0 - skip
        for token_index in token_indices:
            if token_index < 0 or token_index >= num_tokens:
                raise IndexError(f"Token index out of range: {token_index}")
            weights[token_index] += inspect
            counts[token_index] += 1.0
    return np.divide(weights, counts, out=np.zeros_like(weights), where=counts > 0)


def build_skip_hard_chunk_embeddings(
    hidden: np.ndarray, query_weights: np.ndarray, chunks: list[Chunk]
) -> dict[str, np.ndarray]:
    attended = gaze_query_attention(hidden, query_weights)
    embeddings: dict[str, np.ndarray] = {}
    for chunk in chunks:
        if chunk.token_indices is None:
            raise ValueError(f"Chunk {chunk.chunk_id!r} has no token_indices.")
        embeddings[chunk.chunk_id] = pool_chunk_embedding(
            attended, chunk.token_indices, normalize=True
        )
    return embeddings
