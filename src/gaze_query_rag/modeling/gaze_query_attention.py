from __future__ import annotations

import math

import numpy as np

from gaze_query_rag.schemas import Chunk


def _as_hidden(hidden: np.ndarray) -> np.ndarray:
    array = np.asarray(hidden, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("hidden must have shape (tokens, dimensions).")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("hidden must not be empty.")
    return array


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or not np.isfinite(norm):
        return np.zeros_like(vector, dtype=np.float64)
    return vector / norm


def gaze_weight_tokens(hidden: np.ndarray, gaze: np.ndarray) -> np.ndarray:
    hidden_array = _as_hidden(hidden)
    gaze_array = np.asarray(gaze, dtype=np.float64)
    if gaze_array.shape != (hidden_array.shape[0],):
        raise ValueError("gaze must have shape (tokens,).")
    return gaze_array[:, None] * hidden_array


def gaze_query_attention(
    hidden: np.ndarray, gaze: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    hidden_array = _as_hidden(hidden)
    query = gaze_weight_tokens(hidden_array, gaze)
    scores = query @ hidden_array.T / math.sqrt(hidden_array.shape[1])

    valid_mask: np.ndarray | None = None
    if mask is not None:
        valid_mask = np.asarray(mask).astype(bool)
        if valid_mask.shape != (hidden_array.shape[0],):
            raise ValueError("mask must have shape (tokens,).")
        scores[:, ~valid_mask] = -1e30

    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    if valid_mask is not None:
        exp_scores[:, ~valid_mask] = 0.0
    denom = exp_scores.sum(axis=1, keepdims=True)
    probs = np.divide(exp_scores, denom, out=np.zeros_like(exp_scores), where=denom > 0)
    attended = probs @ hidden_array
    if valid_mask is not None:
        attended[~valid_mask] = 0.0
    return attended


def pool_chunk_embedding(
    hidden: np.ndarray, token_indices: list[int], normalize: bool = True
) -> np.ndarray:
    hidden_array = _as_hidden(hidden)
    if not token_indices:
        raise ValueError("token_indices must not be empty.")
    for token_index in token_indices:
        if token_index < 0 or token_index >= hidden_array.shape[0]:
            raise IndexError(f"Token index out of range: {token_index}")
    pooled = hidden_array[np.asarray(token_indices, dtype=np.int64)].mean(axis=0)
    return _normalize(pooled) if normalize else pooled


def _chunk_token_indices(chunk: Chunk) -> list[int]:
    if chunk.token_indices is None:
        raise ValueError(f"Chunk {chunk.chunk_id!r} has no token_indices.")
    return chunk.token_indices


def build_gaze_view_chunk_embeddings(
    hidden: np.ndarray, gaze: np.ndarray, chunks: list[Chunk]
) -> dict[str, np.ndarray]:
    attended = gaze_query_attention(hidden, gaze)
    return {
        chunk.chunk_id: pool_chunk_embedding(attended, _chunk_token_indices(chunk), normalize=True)
        for chunk in chunks
    }


def build_text_chunk_embeddings(hidden: np.ndarray, chunks: list[Chunk]) -> dict[str, np.ndarray]:
    return {
        chunk.chunk_id: pool_chunk_embedding(hidden, _chunk_token_indices(chunk), normalize=True)
        for chunk in chunks
    }
