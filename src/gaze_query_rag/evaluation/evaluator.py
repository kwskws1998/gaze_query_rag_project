from __future__ import annotations

import pandas as pd

from gaze_query_rag.evaluation.metrics import paired_accuracy_delta
from gaze_query_rag.schemas import PredictionRecord


def evaluate_predictions(predictions: list[PredictionRecord]) -> pd.DataFrame:
    rows = []
    for prediction in predictions:
        rows.append(
            {
                "example_id": prediction.example_id,
                "reader_id": prediction.reader_id,
                "condition": prediction.condition,
                "predicted_index": prediction.predicted_index,
                "gold_index": prediction.gold_index,
                "parsed": prediction.predicted_index is not None,
                "has_gold": prediction.gold_index is not None,
                "correct": (
                    prediction.predicted_index == prediction.gold_index
                    if prediction.predicted_index is not None and prediction.gold_index is not None
                    else None
                ),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "condition",
                "reader_id",
                "accuracy",
                "count",
                "scored_count",
                "missing_parse_rate",
            ]
        )
    frame = pd.DataFrame(rows)
    scored = frame[frame["has_gold"]]
    grouped = scored.groupby(["condition", "reader_id"], dropna=False)
    summary = grouped.agg(
        accuracy=("correct", "mean"),
        count=("example_id", "count"),
        scored_count=("correct", lambda values: values.notna().sum()),
        missing_parse_rate=("parsed", lambda values: 1.0 - values.mean()),
    ).reset_index()
    return summary.sort_values(["condition", "reader_id"], na_position="first").reset_index(
        drop=True
    )


def compare_conditions(
    predictions: list[PredictionRecord], baseline: str, treatment: str
) -> dict:
    baseline_predictions = [item for item in predictions if item.condition == baseline]
    treatment_predictions = [item for item in predictions if item.condition == treatment]
    summary = paired_accuracy_delta(treatment_predictions, baseline_predictions)
    return {
        "baseline": baseline,
        "treatment": treatment,
        **summary,
    }
