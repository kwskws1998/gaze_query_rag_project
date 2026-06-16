from __future__ import annotations

import pandas as pd

from gaze_query_rag.evaluation.metrics import retrieval_jaccard
from gaze_query_rag.schemas import PredictionRecord, RetrievalResult


def _prediction_key(prediction: PredictionRecord) -> tuple[str, str | None, str]:
    return (prediction.example_id, prediction.reader_id, prediction.condition)


def _retrieval_key(retrieval: RetrievalResult) -> tuple[str, str | None, str]:
    return (retrieval.example_id, retrieval.reader_id, retrieval.condition)


def analyze_when_gaze_helps(
    predictions: list[PredictionRecord],
    retrievals: list[RetrievalResult],
    gaze_stats: pd.DataFrame,
) -> pd.DataFrame:
    prediction_by_key = {_prediction_key(item): item for item in predictions}
    retrieval_by_key = {_retrieval_key(item): item for item in retrievals}
    text_retrieval_by_example = {
        item.example_id: item for item in retrievals if item.condition == "text"
    }
    gaze_stat_rows = {}
    if not gaze_stats.empty:
        for row in gaze_stats.to_dict(orient="records"):
            gaze_stat_rows[(str(row.get("example_id")), row.get("reader_id"))] = row

    rows = []
    gaze_conditions = ["actual_gaze", "mean_gaze", "shuffled_gaze"]
    for key, prediction in prediction_by_key.items():
        example_id, reader_id, condition = key
        if condition not in gaze_conditions:
            continue
        text_prediction = prediction_by_key.get((example_id, None, "text"))
        if text_prediction is None:
            text_prediction = prediction_by_key.get((example_id, reader_id, "text"))
        if text_prediction is None:
            continue
        if (
            prediction.predicted_index is None
            or prediction.gold_index is None
            or text_prediction.predicted_index is None
            or text_prediction.gold_index is None
        ):
            continue
        treatment_correct = int(prediction.predicted_index == prediction.gold_index)
        text_correct = int(text_prediction.predicted_index == text_prediction.gold_index)
        retrieval = retrieval_by_key.get(key)
        text_retrieval = text_retrieval_by_example.get(example_id)
        row = {
            "example_id": example_id,
            "reader_id": reader_id,
            "condition": condition,
            "delta_vs_text": treatment_correct - text_correct,
            "treatment_correct": treatment_correct,
            "text_correct": text_correct,
            "retrieval_jaccard_vs_text": (
                retrieval_jaccard(retrieval, text_retrieval)
                if retrieval is not None and text_retrieval is not None
                else None
            ),
        }
        row.update(gaze_stat_rows.get((example_id, reader_id), {}))
        rows.append(row)
    return pd.DataFrame(rows)
