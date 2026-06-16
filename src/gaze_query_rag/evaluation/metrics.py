from __future__ import annotations

import math

import numpy as np

from gaze_query_rag.schemas import PredictionRecord, RetrievalResult


def _is_correct(prediction: PredictionRecord) -> bool | None:
    if prediction.predicted_index is None or prediction.gold_index is None:
        return None
    return prediction.predicted_index == prediction.gold_index


def accuracy(predictions: list[PredictionRecord]) -> float:
    scored = [_is_correct(prediction) for prediction in predictions]
    valid = [value for value in scored if value is not None]
    if not valid:
        return float("nan")
    return float(np.mean(valid))


def paired_accuracy_delta(
    a: list[PredictionRecord], b: list[PredictionRecord]
) -> dict[str, float]:
    by_a = {
        (item.example_id, item.reader_id): item
        for item in a
        if item.predicted_index is not None and item.gold_index is not None
    }
    by_b = {
        (item.example_id, item.reader_id): item
        for item in b
        if item.predicted_index is not None and item.gold_index is not None
    }
    deltas = []
    for key in sorted(by_a):
        fallback_key = (key[0], None)
        b_key = key if key in by_b else fallback_key
        if b_key not in by_b:
            continue
        correct_a = 1.0 if by_a[key].predicted_index == by_a[key].gold_index else 0.0
        correct_b = 1.0 if by_b[b_key].predicted_index == by_b[b_key].gold_index else 0.0
        deltas.append(correct_a - correct_b)
    if not deltas:
        return {
            "n": 0,
            "mean_delta": float("nan"),
            "standard_error": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    values = np.asarray(deltas, dtype=np.float64)
    mean = float(values.mean())
    standard_error = (
        float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    )
    return {
        "n": int(len(values)),
        "mean_delta": mean,
        "standard_error": standard_error,
        "ci95_low": mean - 1.96 * standard_error,
        "ci95_high": mean + 1.96 * standard_error,
    }


def retrieval_jaccard(a: RetrievalResult, b: RetrievalResult) -> float:
    set_a = {chunk_id for chunk_id, _ in a.ranked_chunks}
    set_b = {chunk_id for chunk_id, _ in b.ranked_chunks}
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def gaze_concentration(gaze: np.ndarray, top_percent: float) -> float:
    if top_percent <= 0 or top_percent > 1:
        raise ValueError("top_percent must satisfy 0 < top_percent <= 1.")
    values = np.asarray(gaze, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("gaze must be a non-empty one-dimensional array.")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    k = max(1, int(math.ceil(len(values) * top_percent)))
    top = np.sort(values)[-k:]
    return float(top.sum())
