import numpy as np
import pandas as pd

from gaze_query_rag.evaluation.analysis import analyze_when_gaze_helps
from gaze_query_rag.evaluation.evaluator import compare_conditions, evaluate_predictions
from gaze_query_rag.evaluation.metrics import (
    accuracy,
    gaze_concentration,
    paired_accuracy_delta,
    retrieval_jaccard,
)
from gaze_query_rag.schemas import PredictionRecord, RetrievalResult


def _prediction(
    example_id: str,
    condition: str,
    predicted: int | None,
    gold: int = 0,
    reader_id: str | None = None,
) -> PredictionRecord:
    return PredictionRecord(
        example_id=example_id,
        reader_id=reader_id,
        condition=condition,
        prompt="",
        raw_output="",
        predicted_index=predicted,
        gold_index=gold,
    )


def test_accuracy_ignores_unparsed_predictions() -> None:
    predictions = [_prediction("a", "text", 0), _prediction("b", "text", 1), _prediction("c", "text", None)]

    assert accuracy(predictions) == 0.5


def test_paired_accuracy_delta_uses_matched_items() -> None:
    treatment = [_prediction("a", "actual_gaze", 0), _prediction("b", "actual_gaze", 0)]
    baseline = [_prediction("a", "text", 1), _prediction("b", "text", 0)]

    summary = paired_accuracy_delta(treatment, baseline)

    assert summary["n"] == 2
    assert summary["mean_delta"] == 0.5


def test_paired_accuracy_delta_pairs_reader_treatment_with_text_baseline() -> None:
    treatment = [
        _prediction("a", "actual_gaze", 0, reader_id="r1"),
        _prediction("a", "actual_gaze", 1, reader_id="r2"),
    ]
    baseline = [_prediction("a", "text", 1, reader_id=None)]

    summary = paired_accuracy_delta(treatment, baseline)

    assert summary["n"] == 2
    assert summary["mean_delta"] == 0.5


def test_retrieval_jaccard_and_gaze_concentration() -> None:
    a = RetrievalResult("q1", None, "text", [("c1", 1.0), ("c2", 0.5)])
    b = RetrievalResult("q1", "r1", "actual_gaze", [("c2", 0.8), ("c3", 0.7)])

    assert retrieval_jaccard(a, b) == 1 / 3
    assert gaze_concentration(np.array([0.1, 0.2, 0.7]), top_percent=1 / 3) == 0.7


def test_evaluate_predictions_groups_by_condition_and_reader() -> None:
    predictions = [
        _prediction("a", "text", 0),
        _prediction("b", "text", None),
        _prediction("a", "actual_gaze", 0, reader_id="r1"),
    ]

    summary = evaluate_predictions(predictions)

    assert set(summary["condition"]) == {"text", "actual_gaze"}
    text_row = summary[summary["condition"] == "text"].iloc[0]
    assert text_row["missing_parse_rate"] == 0.5


def test_compare_conditions_wraps_paired_delta() -> None:
    predictions = [
        _prediction("a", "text", 1, reader_id=None),
        _prediction("a", "actual_gaze", 0, reader_id="r1"),
    ]

    summary = compare_conditions(predictions, baseline="text", treatment="actual_gaze")

    assert summary["baseline"] == "text"
    assert summary["treatment"] == "actual_gaze"
    assert summary["mean_delta"] == 1.0


def test_analyze_when_gaze_helps_returns_row_level_deltas() -> None:
    predictions = [
        _prediction("a", "text", 1),
        _prediction("a", "actual_gaze", 0, reader_id="r1"),
    ]
    retrievals = [
        RetrievalResult("a", None, "text", [("c1", 1.0)]),
        RetrievalResult("a", "r1", "actual_gaze", [("c2", 1.0)]),
    ]
    gaze_stats = pd.DataFrame([{"example_id": "a", "reader_id": "r1", "concentration": 0.8}])

    rows = analyze_when_gaze_helps(predictions, retrievals, gaze_stats)

    assert len(rows) == 1
    assert rows.iloc[0]["delta_vs_text"] == 1
    assert rows.iloc[0]["retrieval_jaccard_vs_text"] == 0.0
