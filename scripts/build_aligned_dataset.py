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


def _load_matching_gaze_records(
    ia_path: Path,
    target_paragraph_ids: set[str],
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
        filtered = chunk[paragraph_ids.isin(target_paragraph_ids)]
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
    gaze_records = _load_matching_gaze_records(
        args.ia_path,
        target_paragraph_ids=target_paragraph_ids,
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
