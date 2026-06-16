from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from gaze_query_rag.schemas import AlignedExample, GazeRecord, QAExample


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(payload), fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def to_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return to_jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return None if math.isnan(number) else number
    if isinstance(value, float):
        return None if math.isnan(value) else value
    return value


def write_jsonl(path: Path, rows: list[Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            json.dump(to_jsonable(row), fh, ensure_ascii=False)
            fh.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def qa_from_dict(row: dict[str, Any]) -> QAExample:
    return QAExample(
        example_id=str(row["example_id"]),
        paragraph_id=str(row["paragraph_id"]),
        paragraph_text=str(row["paragraph_text"]),
        question=str(row["question"]),
        choices=[str(choice) for choice in row["choices"]],
        answer_index=row.get("answer_index"),
        metadata=dict(row.get("metadata", {})),
    )


def gaze_record_from_dict(row: dict[str, Any]) -> GazeRecord:
    return GazeRecord(
        reader_id=str(row["reader_id"]),
        paragraph_id=str(row["paragraph_id"]),
        word_index=int(row["word_index"]),
        word=str(row["word"]),
        trt=float(row["trt"]),
        fixation_count=None if row.get("fixation_count") is None else float(row["fixation_count"]),
        skip=None if row.get("skip") is None else float(row["skip"]),
        metadata=dict(row.get("metadata", {})),
    )


def aligned_example_from_dict(row: dict[str, Any]) -> AlignedExample:
    return AlignedExample(
        qa=qa_from_dict(row["qa"]),
        reader_gaze={
            str(reader_id): [gaze_record_from_dict(record) for record in records]
            for reader_id, records in row["reader_gaze"].items()
        },
    )


def load_aligned_examples(path: Path) -> list[AlignedExample]:
    return [aligned_example_from_dict(row) for row in read_jsonl(path)]
