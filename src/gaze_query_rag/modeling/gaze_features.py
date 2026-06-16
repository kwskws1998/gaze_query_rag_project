from __future__ import annotations

import numpy as np


def word_trt_to_token_trt(
    word_trt: np.ndarray, word_to_token_indices: list[list[int]], num_tokens: int
) -> np.ndarray:
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive.")
    word_values = np.asarray(word_trt, dtype=np.float64)
    if len(word_values) != len(word_to_token_indices):
        raise ValueError("word_trt and word_to_token_indices must have the same length.")

    token_trt = np.zeros(num_tokens, dtype=np.float64)
    for value, token_indices in zip(word_values, word_to_token_indices):
        if not token_indices:
            continue
        if np.isnan(value):
            value = 0.0
        share = float(value) / len(token_indices)
        for token_index in token_indices:
            if token_index < 0 or token_index >= num_tokens:
                raise IndexError(f"Token index out of range: {token_index}")
            token_trt[token_index] += share
    return token_trt


def compute_gaze_distribution(
    token_trt: np.ndarray, eps: float = 1e-8, transform: str = "log1p"
) -> np.ndarray:
    values = np.asarray(token_trt, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("token_trt must be a one-dimensional array.")
    if len(values) == 0:
        raise ValueError("token_trt must not be empty.")
    if eps <= 0:
        raise ValueError("eps must be positive.")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    if transform == "log1p":
        weights = np.log1p(values)
    elif transform == "identity":
        weights = values.copy()
    elif transform == "none":
        weights = values.copy()
    else:
        raise ValueError(f"Unsupported gaze transform: {transform}")
    weights = weights + eps
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Could not normalize gaze distribution.")
    return weights / total


def compute_mean_gaze(gaze_by_reader: dict[str, np.ndarray]) -> np.ndarray:
    if not gaze_by_reader:
        raise ValueError("gaze_by_reader must not be empty.")
    arrays = [np.asarray(gaze, dtype=np.float64) for gaze in gaze_by_reader.values()]
    lengths = {array.shape for array in arrays}
    if len(lengths) != 1:
        raise ValueError("All gaze vectors must have the same shape.")
    mean = np.mean(np.stack(arrays, axis=0), axis=0)
    total = float(mean.sum())
    if total <= 0 or not np.isfinite(total):
        raise ValueError("Mean gaze cannot be normalized.")
    return mean / total


def shuffle_reader_gaze(gaze_by_reader: dict[str, np.ndarray], seed: int) -> dict[str, np.ndarray]:
    reader_ids = list(gaze_by_reader.keys())
    if not reader_ids:
        raise ValueError("gaze_by_reader must not be empty.")
    if len(reader_ids) == 1:
        return {reader_ids[0]: np.asarray(gaze_by_reader[reader_ids[0]], dtype=np.float64).copy()}

    rng = np.random.default_rng(seed)
    permuted = reader_ids.copy()
    for _ in range(1000):
        rng.shuffle(permuted)
        if all(source != target for source, target in zip(reader_ids, permuted)):
            return {
                target: np.asarray(gaze_by_reader[source], dtype=np.float64).copy()
                for target, source in zip(reader_ids, permuted)
            }
    rotated = reader_ids[1:] + reader_ids[:1]
    return {
        target: np.asarray(gaze_by_reader[source], dtype=np.float64).copy()
        for target, source in zip(reader_ids, rotated)
    }
