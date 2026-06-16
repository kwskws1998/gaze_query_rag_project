from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from gaze_query_rag.schemas import GazeRecord, SchemaInferenceError


REQUIRED_RAW_COLUMNS = {
    "reader_id": "participant_id",
    "article_batch": "article_batch",
    "article_id": "article_id",
    "paragraph_id": "paragraph_id",
    "difficulty_level": "difficulty_level",
    "question_id": "onestopqa_question_id",
    "word_index": "IA_ID",
    "word": "IA_LABEL",
    "trt": "IA_DWELL_TIME",
}


def load_onestop_ia(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    return pd.read_csv(source_path, usecols=columns)


def inspect_onestop_ia_schema(path: str | Path) -> dict[str, Any]:
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    sample = pd.read_csv(source_path, nrows=25)
    row_count = 0
    for chunk in pd.read_csv(source_path, usecols=["participant_id"], chunksize=250_000):
        row_count += len(chunk)
    return {
        "path": str(source_path),
        "row_count": row_count,
        "column_count": len(sample.columns),
        "columns": list(sample.columns),
        "dtypes": {column: str(dtype) for column, dtype in sample.dtypes.items()},
        "sample_rows": sample.to_dict(orient="records"),
    }


def build_gaze_paragraph_id(row: pd.Series) -> str:
    required = ["article_batch", "article_id", "paragraph_id", "difficulty_level"]
    missing = [column for column in required if column not in row.index]
    if missing:
        raise SchemaInferenceError(f"Cannot build gaze paragraph_id; missing columns: {missing}")
    question_id = row.get("onestopqa_question_id", "")
    return (
        f"batch={row['article_batch']}|article={row['article_id']}|paragraph={row['paragraph_id']}"
        f"|level={row['difficulty_level']}|question={question_id}"
    )


def normalize_gaze_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_RAW_COLUMNS.values() if column not in df.columns]
    if missing:
        raise SchemaInferenceError(f"Missing required OneStop IA columns: {missing}")
    out = pd.DataFrame(
        {
            "reader_id": df["participant_id"].astype(str),
            "paragraph_id": df.apply(build_gaze_paragraph_id, axis=1),
            "word_index": pd.to_numeric(df["IA_ID"], errors="raise").astype(int),
            "word": df["IA_LABEL"].astype(str),
            "trt": pd.to_numeric(df["IA_DWELL_TIME"], errors="coerce").fillna(0.0),
        }
    )
    out["fixation_count"] = (
        pd.to_numeric(df["IA_FIXATION_COUNT"], errors="coerce")
        if "IA_FIXATION_COUNT" in df.columns
        else None
    )
    out["skip"] = (
        pd.to_numeric(df["IA_SKIP"], errors="coerce") if "IA_SKIP" in df.columns else None
    )
    metadata_columns = [
        column
        for column in [
            "article_batch",
            "article_id",
            "paragraph_id",
            "difficulty_level",
            "onestopqa_question_id",
            "question_preview",
            "repeated_reading_trial",
            "practice_trial",
            "trial_index",
            "TRIAL_INDEX",
            "paragraph",
            "question",
            "selected_answer",
            "is_correct",
        ]
        if column in df.columns
    ]
    for column in metadata_columns:
        out[f"metadata__{column}"] = df[column]
    return out


def to_gaze_records(df: pd.DataFrame) -> list[GazeRecord]:
    required = ["reader_id", "paragraph_id", "word_index", "word", "trt"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise SchemaInferenceError(f"Missing canonical gaze columns: {missing}")
    records: list[GazeRecord] = []
    metadata_cols = [column for column in df.columns if column.startswith("metadata__")]
    for row in df.itertuples(index=False):
        values = row._asdict()
        metadata = {column.removeprefix("metadata__"): values[column] for column in metadata_cols}
        fixation_count = values.get("fixation_count")
        skip = values.get("skip")
        records.append(
            GazeRecord(
                reader_id=str(values["reader_id"]),
                paragraph_id=str(values["paragraph_id"]),
                word_index=int(values["word_index"]),
                word=str(values["word"]),
                trt=float(values["trt"]),
                fixation_count=None if pd.isna(fixation_count) else float(fixation_count),
                skip=None if pd.isna(skip) else float(skip),
                metadata=metadata,
            )
        )
    return records
