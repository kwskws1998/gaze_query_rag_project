from __future__ import annotations

import re
from collections import Counter, defaultdict

from gaze_query_rag.schemas import AlignedExample, GazeRecord, QAExample


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _sorted_reader_gaze(records: list[GazeRecord]) -> dict[str, list[GazeRecord]]:
    by_reader: dict[str, list[GazeRecord]] = defaultdict(list)
    for record in records:
        by_reader[record.reader_id].append(record)
    return {
        reader_id: sorted(reader_records, key=lambda item: item.word_index)
        for reader_id, reader_records in sorted(by_reader.items())
    }


def _group_gaze_by_paragraph(
    gaze_records: list[GazeRecord],
) -> dict[str, dict[str, list[GazeRecord]]]:
    grouped: dict[str, list[GazeRecord]] = defaultdict(list)
    for record in gaze_records:
        grouped[record.paragraph_id].append(record)
    return {paragraph_id: _sorted_reader_gaze(records) for paragraph_id, records in grouped.items()}


def _group_gaze_by_text(
    gaze_records: list[GazeRecord],
) -> tuple[
    dict[tuple[str, str], dict[str, list[GazeRecord]]],
    dict[str, dict[str, list[GazeRecord]]],
]:
    by_text_question: dict[tuple[str, str], list[GazeRecord]] = defaultdict(list)
    by_text: dict[str, list[GazeRecord]] = defaultdict(list)
    for record in gaze_records:
        paragraph = record.metadata.get("paragraph")
        if not paragraph:
            continue
        paragraph_key = _normalize_text(str(paragraph))
        question = record.metadata.get("question")
        if question:
            by_text_question[(paragraph_key, _normalize_text(str(question)))].append(record)
        by_text[paragraph_key].append(record)
    return (
        {key: _sorted_reader_gaze(records) for key, records in by_text_question.items()},
        {key: _sorted_reader_gaze(records) for key, records in by_text.items()},
    )


def align_qa_with_gaze(
    qa_examples: list[QAExample], gaze_records: list[GazeRecord]
) -> list[AlignedExample]:
    by_paragraph = _group_gaze_by_paragraph(gaze_records)
    by_text_question, by_text = _group_gaze_by_text(gaze_records)

    aligned: list[AlignedExample] = []
    for qa in qa_examples:
        reader_gaze = by_paragraph.get(qa.paragraph_id)
        if reader_gaze is None:
            text_key = _normalize_text(qa.paragraph_text)
            question_key = _normalize_text(qa.question)
            reader_gaze = by_text_question.get((text_key, question_key))
        if reader_gaze is None:
            reader_gaze = by_text.get(_normalize_text(qa.paragraph_text))
        if reader_gaze:
            aligned.append(AlignedExample(qa=qa, reader_gaze=reader_gaze))
    return aligned


def build_alignment_report(
    aligned: list[AlignedExample],
    qa_examples: list[QAExample],
    gaze_records: list[GazeRecord],
) -> dict:
    matched_example_ids = {item.qa.example_id for item in aligned}
    matched_gaze_paragraph_ids = {
        record.paragraph_id
        for item in aligned
        for records in item.reader_gaze.values()
        for record in records
    }
    qa_paragraph_ids = {item.paragraph_id for item in qa_examples}
    gaze_paragraph_ids = {item.paragraph_id for item in gaze_records}
    reader_counts = [len(item.reader_gaze) for item in aligned]
    record_counts = [sum(len(records) for records in item.reader_gaze.values()) for item in aligned]
    all_reader_ids = {record.reader_id for record in gaze_records}
    matched_reader_ids = {
        reader_id for item in aligned for reader_id in item.reader_gaze.keys()
    }
    method_counts = Counter(
        "exact_id"
        if any(
            record.paragraph_id == item.qa.paragraph_id
            for records in item.reader_gaze.values()
            for record in records[:1]
        )
        else "text_or_metadata"
        for item in aligned
    )
    return {
        "qa_examples": len(qa_examples),
        "gaze_records": len(gaze_records),
        "aligned_examples": len(aligned),
        "unmatched_qa_examples": len(qa_examples) - len(matched_example_ids),
        "qa_paragraphs": len(qa_paragraph_ids),
        "gaze_paragraphs": len(gaze_paragraph_ids),
        "matched_gaze_paragraphs": len(matched_gaze_paragraph_ids),
        "paragraph_coverage": {
            "qa_matched_fraction": (
                len({item.qa.paragraph_id for item in aligned}) / len(qa_paragraph_ids)
                if qa_paragraph_ids
                else 0.0
            ),
            "gaze_matched_fraction": (
                len(matched_gaze_paragraph_ids) / len(gaze_paragraph_ids)
                if gaze_paragraph_ids
                else 0.0
            ),
        },
        "reader_coverage": {
            "total_readers": len(all_reader_ids),
            "matched_readers": len(matched_reader_ids),
            "matched_fraction": (
                len(matched_reader_ids) / len(all_reader_ids) if all_reader_ids else 0.0
            ),
        },
        "readers_per_aligned_example": {
            "min": min(reader_counts) if reader_counts else 0,
            "max": max(reader_counts) if reader_counts else 0,
            "mean": sum(reader_counts) / len(reader_counts) if reader_counts else 0.0,
        },
        "gaze_records_per_aligned_example": {
            "min": min(record_counts) if record_counts else 0,
            "max": max(record_counts) if record_counts else 0,
            "mean": sum(record_counts) / len(record_counts) if record_counts else 0.0,
        },
        "alignment_methods": dict(method_counts),
        "unmatched_qa_example_ids": [
            item.example_id for item in qa_examples if item.example_id not in matched_example_ids
        ],
        "unmatched_gaze_paragraph_ids": sorted(gaze_paragraph_ids - matched_gaze_paragraph_ids),
    }
