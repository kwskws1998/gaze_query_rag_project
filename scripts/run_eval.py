#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gaze_query_rag.evaluation.evaluator import compare_conditions, evaluate_predictions
from gaze_query_rag.evaluation.metrics import retrieval_jaccard
from gaze_query_rag.io.artifacts import read_jsonl, write_json
from gaze_query_rag.schemas import PredictionRecord, RetrievalResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-path", type=Path, default=ROOT / "artifacts/predictions/predictions.jsonl")
    parser.add_argument("--retrieval-path", type=Path, default=ROOT / "artifacts/retrieval/retrieval_results.jsonl")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--baseline-condition", default=None)
    return parser.parse_args()


def _prediction_from_row(row: dict) -> PredictionRecord:
    return PredictionRecord(
        example_id=str(row["example_id"]),
        reader_id=row.get("reader_id"),
        condition=str(row["condition"]),
        prompt=str(row.get("prompt", "")),
        raw_output=str(row.get("raw_output", "")),
        predicted_index=row.get("predicted_index"),
        gold_index=row.get("gold_index"),
        metadata=dict(row.get("metadata", {})),
    )


def _retrieval_from_row(row: dict) -> RetrievalResult:
    return RetrievalResult(
        example_id=str(row["example_id"]),
        reader_id=row.get("reader_id"),
        condition=str(row["condition"]),
        ranked_chunks=[(str(chunk_id), float(score)) for chunk_id, score in row["ranked_chunks"]],
        metadata=dict(row.get("metadata", {})),
    )


def _retrieval_overlap_frame(retrievals: list[RetrievalResult]) -> pd.DataFrame:
    text_by_example = {item.example_id: item for item in retrievals if item.condition == "text"}
    rows = []
    for item in retrievals:
        if item.condition == "text":
            continue
        text = text_by_example.get(item.example_id)
        rows.append(
            {
                "example_id": item.example_id,
                "reader_id": item.reader_id,
                "condition": item.condition,
                "jaccard_vs_text": retrieval_jaccard(item, text) if text is not None else None,
                "top_chunk": item.ranked_chunks[0][0] if item.ranked_chunks else None,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    results_dir = args.artifacts_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    predictions: list[PredictionRecord] = []
    if args.predictions_path.exists():
        predictions = [_prediction_from_row(row) for row in read_jsonl(args.predictions_path)]

    retrievals: list[RetrievalResult] = []
    if args.retrieval_path.exists():
        retrievals = [_retrieval_from_row(row) for row in read_jsonl(args.retrieval_path)]

    by_condition_path = results_dir / "by_condition.csv"
    paired_path = results_dir / "paired_deltas.csv"
    if predictions:
        by_condition = evaluate_predictions(predictions)
        by_condition.to_csv(by_condition_path, index=False)
        paired_rows = []
        conditions = sorted({item.condition for item in predictions})
        baseline = args.baseline_condition
        if baseline is None:
            baseline = "text" if "text" in conditions else "bare_model"
        for condition in conditions:
            if condition == baseline:
                continue
            paired_rows.append(compare_conditions(predictions, baseline=baseline, treatment=condition))
        pd.DataFrame(paired_rows).to_csv(paired_path, index=False)
    else:
        pd.DataFrame(columns=["condition", "reader_id", "accuracy", "count", "scored_count", "missing_parse_rate"]).to_csv(
            by_condition_path, index=False
        )
        pd.DataFrame(columns=["baseline", "treatment", "n", "mean_delta", "standard_error", "ci95_low", "ci95_high"]).to_csv(
            paired_path, index=False
        )

    retrieval_overlap = _retrieval_overlap_frame(retrievals)
    retrieval_overlap_path = results_dir / "retrieval_overlap.csv"
    retrieval_overlap.to_csv(retrieval_overlap_path, index=False)
    overlap_by_condition = {}
    if not retrieval_overlap.empty:
        overlap_by_condition = (
            retrieval_overlap.groupby("condition")["jaccard_vs_text"]
            .agg(["count", "mean", "min", "max"])
            .reset_index()
            .to_dict(orient="records")
        )

    retrieval_counts = defaultdict(int)
    for item in retrievals:
        retrieval_counts[item.condition] += 1
    summary = {
        "prediction_records": len(predictions),
        "retrieval_records": len(retrievals),
        "retrieval_records_by_condition": dict(sorted(retrieval_counts.items())),
        "overlap_by_condition": overlap_by_condition,
        "by_condition_path": str(by_condition_path),
        "paired_deltas_path": str(paired_path),
        "retrieval_overlap_path": str(retrieval_overlap_path),
    }
    write_json(results_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
