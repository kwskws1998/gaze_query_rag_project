from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np


FULL_CONTEXT = "full_context"
QUERY_ONLY_COMPRESSION = "query_only_compression"
PREDICTED_ET_ONLY_COMPRESSION = "predicted_et_only_compression"
QUERY_X_PREDICTED_ET_COMPRESSION = "query_x_predicted_et_compression"
COMPRESSION_CONDITIONS = {
    QUERY_ONLY_COMPRESSION,
    PREDICTED_ET_ONLY_COMPRESSION,
    QUERY_X_PREDICTED_ET_COMPRESSION,
}
SUPPORTED_CONDITIONS = {FULL_CONTEXT, *COMPRESSION_CONDITIONS}


@dataclass(frozen=True)
class ContextChunk:
    chunk_id: str
    text: str
    char_start: int
    char_end: int
    index: int


@dataclass(frozen=True)
class CompressedContext:
    text: str
    selected_chunks: list[ContextChunk]
    selected_scores: list[float]
    original_token_count: int
    compressed_token_count: int

    @property
    def compression_ratio(self) -> float:
        if self.original_token_count <= 0:
            return 1.0
        return self.compressed_token_count / self.original_token_count


def split_context_chunks(text: str, max_words: int = 80) -> list[ContextChunk]:
    if max_words <= 0:
        raise ValueError("max_words must be positive.")
    chunks: list[ContextChunk] = []
    for match in re.finditer(r"[^\n.!?]+(?:[.!?]+|$)", text):
        sentence = match.group(0).strip()
        if not sentence:
            continue
        start = match.start() + len(match.group(0)) - len(match.group(0).lstrip())
        end = start + len(sentence)
        words = list(re.finditer(r"\S+", sentence))
        if len(words) <= max_words:
            chunks.append(_make_chunk(sentence, start, end, len(chunks)))
            continue
        for word_start in range(0, len(words), max_words):
            word_group = words[word_start : word_start + max_words]
            local_start = word_group[0].start()
            local_end = word_group[-1].end()
            chunk_text = sentence[local_start:local_end]
            chunks.append(
                _make_chunk(
                    chunk_text,
                    start + local_start,
                    start + local_end,
                    len(chunks),
                )
            )
    if chunks:
        return chunks
    stripped = text.strip()
    if not stripped:
        raise ValueError("Cannot split an empty context.")
    start = text.index(stripped)
    return [_make_chunk(stripped, start, start + len(stripped), 0)]


def predicted_trt_chunk_scores(predictor, chunks: list[ContextChunk]) -> np.ndarray:
    scores: list[float] = []
    for chunk in chunks:
        predicted_words = predictor.predict_words(chunk.text)
        trt = np.asarray([max(0.0, row.trt) for row in predicted_words], dtype=np.float64)
        if trt.size == 0:
            scores.append(0.0)
        else:
            scores.append(float(np.mean(np.log1p(trt))))
    return np.asarray(scores, dtype=np.float64)


def combine_chunk_scores(
    condition: str,
    query_scores: np.ndarray | None = None,
    trt_scores: np.ndarray | None = None,
    alpha: float = 1.0,
    beta: float = 0.25,
    tau_q: float = 0.05,
    tau_g: float = 0.1,
    eps: float = 1e-12,
) -> np.ndarray:
    if condition == QUERY_ONLY_COMPRESSION:
        if query_scores is None:
            raise ValueError("query_scores are required for query-only compression.")
        return _softmax(query_scores, tau_q)
    if condition == PREDICTED_ET_ONLY_COMPRESSION:
        if trt_scores is None:
            raise ValueError("trt_scores are required for predicted-ET compression.")
        return _softmax(trt_scores, tau_g)
    if condition == QUERY_X_PREDICTED_ET_COMPRESSION:
        if query_scores is None or trt_scores is None:
            raise ValueError("query_scores and trt_scores are required for query x predicted-ET compression.")
        query_distribution = _softmax(query_scores, tau_q)
        trt_distribution = _softmax(trt_scores, tau_g)
        combined = np.power(query_distribution + eps, alpha) * np.power(trt_distribution + eps, beta)
        total = float(np.sum(combined))
        if total <= 0.0 or not np.isfinite(total):
            return np.full_like(combined, 1.0 / len(combined), dtype=np.float64)
        return combined / total
    raise ValueError(f"Unsupported compression condition: {condition}")


def compress_context_by_scores(
    text: str,
    chunks: list[ContextChunk],
    scores: np.ndarray,
    budget_tokens: int,
    token_count_fn: Callable[[str], int],
) -> CompressedContext:
    if budget_tokens <= 0:
        raise ValueError("budget_tokens must be positive.")
    if len(chunks) == 0:
        raise ValueError("chunks must not be empty.")
    if len(scores) != len(chunks):
        raise ValueError("scores must have the same length as chunks.")

    original_token_count = token_count_fn(text)
    token_counts = [max(1, token_count_fn(chunk.text)) for chunk in chunks]
    ranked = sorted(
        range(len(chunks)),
        key=lambda index: (-float(scores[index]), chunks[index].index),
    )
    selected_indices: list[int] = []
    total_tokens = 0
    for index in ranked:
        next_total = total_tokens + token_counts[index]
        if next_total <= budget_tokens or not selected_indices:
            selected_indices.append(index)
            total_tokens = next_total
    selected_indices.sort(key=lambda index: chunks[index].index)
    selected_chunks = [chunks[index] for index in selected_indices]
    selected_scores = [float(scores[index]) for index in selected_indices]
    compressed_text = "\n\n".join(chunk.text for chunk in selected_chunks)
    return CompressedContext(
        text=compressed_text,
        selected_chunks=selected_chunks,
        selected_scores=selected_scores,
        original_token_count=original_token_count,
        compressed_token_count=token_count_fn(compressed_text),
    )


def _make_chunk(text: str, start: int, end: int, index: int) -> ContextChunk:
    return ContextChunk(
        chunk_id=f"context:chunk={index}",
        text=text,
        char_start=start,
        char_end=end,
        index=index,
    )


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError("values must be a non-empty 1D array.")
    safe = np.where(np.isfinite(array), array, -np.inf)
    if not np.any(np.isfinite(safe)):
        return np.full(len(array), 1.0 / len(array), dtype=np.float64)
    scaled = safe / temperature
    scaled = scaled - np.max(scaled)
    weights = np.exp(scaled)
    total = float(np.sum(weights))
    if total <= 0.0 or not np.isfinite(total):
        return np.full(len(array), 1.0 / len(array), dtype=np.float64)
    return weights / total
