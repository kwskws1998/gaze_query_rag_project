#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gaze_query_rag.data.alignment import align_qa_with_gaze, build_alignment_report
from gaze_query_rag.data.hf_onestop_qa import load_local_onestop_qa_json, load_onestop_qa
from gaze_query_rag.data.onestop_gaze import normalize_gaze_schema, to_gaze_records
from gaze_query_rag.io.artifacts import write_json, write_jsonl
from gaze_query_rag.schemas import AlignedExample, GazeRecord, QAExample


IA_COLUMNS = [
    "participant_id",
    "article_batch",
    "article_id",
    "article_title",
    "paragraph_id",
    "difficulty_level",
    "onestopqa_question_id",
    "IA_ID",
    "IA_LABEL",
    "IA_DWELL_TIME",
    "IA_FIXATION_COUNT",
    "IA_SKIP",
    "TRIAL_INDEX",
    "trial_index",
    "question_preview",
    "repeated_reading_trial",
    "practice_trial",
    "paragraph",
    "question",
    "selected_answer",
    "is_correct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qa-json-path",
        type=Path,
        default=ROOT / "OneStop-Eye-Movements/data_preprocessing/onestop_qa.json",
    )
    parser.add_argument("--qa-dataset-name", default="malmaud/onestop_qa")
    parser.add_argument("--qa-split", default=None)
    parser.add_argument(
        "--ia-path",
        type=Path,
        default=ROOT
        / "OneStop-Eye-Movements/data/OneStop/osfstorage-archive (1)/ia_Paragraph.csv.zip",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-readers-per-example", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--use-hf", action="store_true")
    parser.add_argument("--include-practice", action="store_true")
    parser.add_argument("--exclude-repeated", action="store_true")
    return parser.parse_args()


def _limit_examples(qa_examples: list[QAExample], max_examples: int | None) -> list[QAExample]:
    if max_examples is None:
        return qa_examples
    if max_examples <= 0:
        raise ValueError("max_examples must be positive when provided.")
    return qa_examples[:max_examples]


def _paragraph_ids_for_chunk(chunk: pd.DataFrame) -> pd.Series:
    return (
        "batch="
        + chunk["article_batch"].astype(str)
        + "|article="
        + chunk["article_id"].astype(str)
        + "|paragraph="
        + chunk["paragraph_id"].astype(str)
        + "|level="
        + chunk["difficulty_level"].astype(str)
        + "|question="
        + chunk["onestopqa_question_id"].astype(str)
    )


def _normalize_match_text(text: str) -> str:
    return " ".join(str(text).split()).strip().lower()


def _normalized_text_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip().str.lower()


def _normalize_level(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return {0: "adv", 1: "int", 2: "ele"}.get(value, str(value).lower())
    text = str(value).strip().lower()
    if text.isdigit():
        return {0: "adv", 1: "int", 2: "ele"}.get(int(text), text)
    if text.startswith("adv"):
        return "adv"
    if text.startswith("int"):
        return "int"
    if text.startswith("ele"):
        return "ele"
    return text


def _qa_metadata_key(qa: QAExample) -> tuple[str, str, str] | None:
    raw = qa.metadata.get("raw") if isinstance(qa.metadata.get("raw"), dict) else {}
    title = raw["title"] if "title" in raw else qa.metadata.get("article_title")
    level = raw["level"] if "level" in raw else qa.metadata.get("difficulty_level")
    if title is None or level is None:
        return None
    return (
        _normalize_match_text(str(title)),
        _normalize_level(level),
        _normalize_match_text(qa.question),
    )


def _load_matching_gaze_records(
    ia_path: Path,
    target_paragraph_ids: set[str],
    target_paragraph_texts: set[str],
    target_metadata_keys: set[tuple[str, str, str]],
    chunksize: int,
    include_practice: bool,
    exclude_repeated: bool,
) -> list[GazeRecord]:
    records: list[GazeRecord] = []
    for chunk in pd.read_csv(ia_path, usecols=IA_COLUMNS, chunksize=chunksize):
        if not include_practice and "practice_trial" in chunk.columns:
            chunk = chunk[chunk["practice_trial"] == False]
        if exclude_repeated and "repeated_reading_trial" in chunk.columns:
            chunk = chunk[chunk["repeated_reading_trial"] == False]
        if chunk.empty:
            continue
        paragraph_ids = _paragraph_ids_for_chunk(chunk)
        id_mask = paragraph_ids.isin(target_paragraph_ids)
        metadata_mask = pd.Series(False, index=chunk.index)
        if target_metadata_keys:
            title_values = _normalized_text_series(chunk["article_title"])
            level_values = chunk["difficulty_level"].map(_normalize_level)
            question_values = _normalized_text_series(chunk["question"])
            metadata_mask = pd.Series(
                [
                    (title, level, question) in target_metadata_keys
                    for title, level, question in zip(title_values, level_values, question_values)
                ],
                index=chunk.index,
            )
        if target_paragraph_texts:
            text_mask = _normalized_text_series(chunk["paragraph"]).isin(target_paragraph_texts)
            filtered = chunk[id_mask | text_mask | metadata_mask]
        else:
            filtered = chunk[id_mask | metadata_mask]
        if filtered.empty:
            continue
        canonical = normalize_gaze_schema(filtered)
        records.extend(to_gaze_records(canonical))
    return records


def _trim_readers(
    aligned: list[AlignedExample], max_readers_per_example: int | None
) -> list[AlignedExample]:
    if max_readers_per_example is None:
        return aligned
    if max_readers_per_example <= 0:
        raise ValueError("max_readers_per_example must be positive when provided.")
    trimmed: list[AlignedExample] = []
    for item in aligned:
        reader_ids = sorted(item.reader_gaze)[:max_readers_per_example]
        trimmed.append(
            AlignedExample(
                qa=item.qa,
                reader_gaze={reader_id: item.reader_gaze[reader_id] for reader_id in reader_ids},
            )
        )
    return trimmed


def _qa_to_dict(qa: QAExample) -> dict:
    return {
        "example_id": qa.example_id,
        "paragraph_id": qa.paragraph_id,
        "paragraph_text": qa.paragraph_text,
        "question": qa.question,
        "choices": qa.choices,
        "answer_index": qa.answer_index,
        "metadata": qa.metadata,
    }


def _gaze_to_dict(record: GazeRecord) -> dict:
    return {
        "reader_id": record.reader_id,
        "paragraph_id": record.paragraph_id,
        "word_index": record.word_index,
        "word": record.word,
        "trt": record.trt,
        "fixation_count": record.fixation_count,
        "skip": record.skip,
        "metadata": record.metadata,
    }


def _aligned_to_dict(item: AlignedExample) -> dict:
    return {
        "qa": _qa_to_dict(item.qa),
        "reader_gaze": {
            reader_id: [_gaze_to_dict(record) for record in records]
            for reader_id, records in item.reader_gaze.items()
        },
    }


def main() -> None:
    args = parse_args()
    if args.use_hf:
        qa_examples = load_onestop_qa(args.qa_dataset_name, args.qa_split, cache_dir=None)
    else:
        qa_examples = load_local_onestop_qa_json(args.qa_json_path)
    qa_examples = _limit_examples(qa_examples, args.max_examples)
    target_paragraph_ids = {qa.paragraph_id for qa in qa_examples}
    target_paragraph_texts = {_normalize_match_text(qa.paragraph_text) for qa in qa_examples}
    target_metadata_keys = {key for qa in qa_examples if (key := _qa_metadata_key(qa)) is not None}
    gaze_records = _load_matching_gaze_records(
        args.ia_path,
        target_paragraph_ids=target_paragraph_ids,
        target_paragraph_texts=target_paragraph_texts,
        target_metadata_keys=target_metadata_keys,
        chunksize=args.chunksize,
        include_practice=args.include_practice,
        exclude_repeated=args.exclude_repeated,
    )
    aligned = align_qa_with_gaze(qa_examples, gaze_records)
    aligned = _trim_readers(aligned, args.max_readers_per_example)

    trimmed_gaze_records = [
        record for item in aligned for records in item.reader_gaze.values() for record in records
    ]
    report = build_alignment_report(aligned, qa_examples, trimmed_gaze_records)
    report.update(
        {
            "ia_path": str(args.ia_path),
            "qa_json_path": str(args.qa_json_path),
            "max_examples": args.max_examples,
            "max_readers_per_example": args.max_readers_per_example,
            "include_practice": args.include_practice,
            "exclude_repeated": args.exclude_repeated,
        }
    )

    data_dir = args.artifacts_dir / "data"
    write_jsonl(data_dir / "aligned_examples.jsonl", [_aligned_to_dict(item) for item in aligned])
    write_json(data_dir / "alignment_report.json", report)

    print(
        json.dumps(
            {
                "qa_examples": len(qa_examples),
                "gaze_records": len(trimmed_gaze_records),
                "aligned_examples": len(aligned),
                "alignment_report": str(data_dir / "alignment_report.json"),
                "aligned_examples_path": str(data_dir / "aligned_examples.jsonl"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
