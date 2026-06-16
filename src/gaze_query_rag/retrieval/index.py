from __future__ import annotations

import numpy as np

from gaze_query_rag.schemas import EmbeddingIndex


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix, dtype=np.float64), where=norms > 0)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or not np.isfinite(norm):
        return np.zeros_like(vector, dtype=np.float64)
    return vector / norm


def build_in_memory_index(chunk_embeddings: dict[str, np.ndarray]) -> EmbeddingIndex:
    if not chunk_embeddings:
        raise ValueError("chunk_embeddings must not be empty.")
    ids = sorted(chunk_embeddings)
    vectors = [np.asarray(chunk_embeddings[chunk_id], dtype=np.float64) for chunk_id in ids]
    shapes = {vector.shape for vector in vectors}
    if len(shapes) != 1:
        raise ValueError("All chunk embeddings must have the same shape.")
    matrix = _normalize_matrix(np.stack(vectors, axis=0))
    return EmbeddingIndex(ids=ids, matrix=matrix)


def search_index(index: EmbeddingIndex, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    query_vector = _normalize_vector(np.asarray(query, dtype=np.float64))
    if index.matrix.ndim != 2:
        raise ValueError("index matrix must have shape (items, dimensions).")
    if query_vector.shape != (index.matrix.shape[1],):
        raise ValueError("query dimension does not match index dimension.")
    scores = index.matrix @ query_vector
    order = np.argsort(-scores, kind="mergesort")[: min(top_k, len(index.ids))]
    return [(index.ids[int(idx)], float(scores[int(idx)])) for idx in order]
