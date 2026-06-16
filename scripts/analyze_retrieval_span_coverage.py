#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gaze_query_rag.io.artifacts import write_json


CONDITION_FILES = {
    "text": "text_chunks.npz",
    "actual_gaze": "gaze_chunks_actual.npz",
    "mean_gaze": "gaze_chunks_mean.npz",
    "shuffled_gaze": "gaze_chunks_shuffled.npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-path", type=Path, required=True)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--qa-json-path", type=Path, default=ROOT / "resources/onestop_qa.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/results")
    parser.add_argument("--conditions", nargs="+", default=["text", "actual_gaze"])
    parser.add_argument("--top-k-values", nargs="+", type=int, default=[1, 3])
    return parser.parse_args()


def _word_char_spans(text: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def _span_words_to_chars(
    word_spans: list[tuple[int, int]], span_indices: list[int]
) -> list[tuple[int, int]]:
    char_spans: list[tuple[int, int]] = []
    if len(span_indices) % 2 != 0:
        raise ValueError(f"Span index list must contain start/end pairs: {span_indices}")
    for start_word, end_word in zip(span_indices[0::2], span_indices[1::2]):
        if start_word < 0 or end_word >= len(word_spans) or start_word > end_word:
            raise ValueError(
                f"Invalid word span {start_word}:{end_word} for {len(word_spans)} words."
            )
        char_spans.append((word_spans[start_word][0], word_spans[end_word][1]))
    return char_spans


def _load_local_qa_spans(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    articles = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    spans: dict[str, dict[str, Any]] = {}
    for article in articles:
        article_id = str(article["article_id"])
        for paragraph in article.get("paragraphs", []):
            paragraph_id = paragraph.get("paragraph_id")
            qas = paragraph.get("qas", [])
            for level, level_payload in paragraph.items():
                if level in {"paragraph_id", "qas"} or not isinstance(level_payload, dict):
                    continue
                context = str(level_payload.get("context", ""))
                word_spans = _word_char_spans(context)
                a_spans = level_payload.get("a_spans", [])
                d_spans = level_payload.get("d_spans", [])
                for qa in qas:
                    q_ind = int(qa["q_ind"])
                    example_id = f"local:{article_id}:{paragraph_id}:{level}:{q_ind}"
                    spans[example_id] = {
                        "paragraph_text": context,
                        "level": level,
                        "question": str(qa["question"]),
                        "a_char_spans": _span_words_to_chars(word_spans, a_spans[q_ind]),
                        "d_char_spans": _span_words_to_chars(word_spans, d_spans[q_ind]),
                    }
    return spans


def _load_embedding_records(path: Path) -> dict[tuple[str, str, str | None, str], dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    records = json.loads(str(data["records_json"].item()))
    out: dict[tuple[str, str, str | None, str], dict[str, Any]] = {}
    for record in records:
        key = (
            str(record["condition"]),
            str(record["example_id"]),
            record.get("reader_id"),
            str(record["chunk_id"]),
        )
        out[key] = record
    return out


def _load_records_for_conditions(
    embeddings_dir: Path, conditions: list[str]
) -> dict[tuple[str, str, str | None, str], dict[str, Any]]:
    records: dict[tuple[str, str, str | None, str], dict[str, Any]] = {}
    for condition in conditions:
        filename = CONDITION_FILES.get(condition)
        if filename is None:
            raise ValueError(f"Unsupported condition: {condition}")
        path = embeddings_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        records.update(_load_embedding_records(path))
    return records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _interval_overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[1] > b[0] and b[1] > a[0]


def _interval_contains(container: tuple[int, int], target: tuple[int, int]) -> bool:
    return container[0] <= target[0] and container[1] >= target[1]


def _coverage_flags(
    retrieved_intervals: list[tuple[int, int]], target_spans: list[tuple[int, int]]
) -> dict[str, bool]:
    any_overlap = any(
        _interval_overlaps(interval, span)
        for interval in retrieved_intervals
        for span in target_spans
    )
    any_full = any(
        _interval_contains(interval, span)
        for interval in retrieved_intervals
        for span in target_spans
    )
    all_full = all(
        any(_interval_contains(interval, span) for interval in retrieved_intervals)
        for span in target_spans
    )
    return {
        "any_overlap": any_overlap,
        "any_full": any_full,
        "all_full": all_full,
    }


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if any(top_k <= 0 for top_k in args.top_k_values):
        raise ValueError("--top-k-values must be positive.")

    qa_spans = _load_local_qa_spans(args.qa_json_path)
    chunk_records = _load_records_for_conditions(args.embeddings_dir, args.conditions)
    retrieval_rows = [
        row for row in _read_jsonl(args.retrieval_path) if row.get("condition") in set(args.conditions)
    ]
    if not retrieval_rows:
        raise ValueError(f"No matching retrieval rows found at {args.retrieval_path}.")

    detail_rows: list[dict[str, Any]] = []
    for row in retrieval_rows:
        example_id = str(row["example_id"])
        condition = str(row["condition"])
        reader_id = row.get("reader_id")
        if example_id not in qa_spans:
            raise KeyError(f"Missing QA spans for example {example_id!r}.")
        ranked_chunks = [str(chunk_id) for chunk_id, _ in row["ranked_chunks"]]
        max_available = len(ranked_chunks)
        for top_k in args.top_k_values:
            selected_chunks = ranked_chunks[:top_k]
            intervals: list[tuple[int, int]] = []
            for chunk_id in selected_chunks:
                record = chunk_records.get((condition, example_id, reader_id, chunk_id))
                if record is None:
                    raise KeyError(
                        f"Missing chunk record for {(condition, example_id, reader_id, chunk_id)!r}"
                    )
                intervals.append((int(record["char_start"]), int(record["char_end"])))
            a_flags = _coverage_flags(intervals, qa_spans[example_id]["a_char_spans"])
            d_flags = _coverage_flags(intervals, qa_spans[example_id]["d_char_spans"])
            detail_rows.append(
                {
                    "example_id": example_id,
                    "reader_id": reader_id,
                    "condition": condition,
                    "level": qa_spans[example_id]["level"],
                    "top_k": top_k,
                    "available_ranked_chunks": max_available,
                    "a_any_overlap": int(a_flags["any_overlap"]),
                    "a_any_full": int(a_flags["any_full"]),
                    "a_all_full": int(a_flags["all_full"]),
                    "d_any_overlap": int(d_flags["any_overlap"]),
                    "d_any_full": int(d_flags["any_full"]),
                    "d_all_full": int(d_flags["all_full"]),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        grouped[(str(row["condition"]), int(row["top_k"]))].append(row)
    for (condition, top_k), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "condition": condition,
                "top_k": top_k,
                "count": len(rows),
                "a_any_overlap": _mean([row["a_any_overlap"] for row in rows]),
                "a_any_full": _mean([row["a_any_full"] for row in rows]),
                "a_all_full": _mean([row["a_all_full"] for row in rows]),
                "d_any_overlap": _mean([row["d_any_overlap"] for row in rows]),
                "d_any_full": _mean([row["d_any_full"] for row in rows]),
                "d_all_full": _mean([row["d_all_full"] for row in rows]),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "retrieval_span_coverage_detail.csv"
    summary_path = args.output_dir / "retrieval_span_coverage_summary.csv"
    json_path = args.output_dir / "retrieval_span_coverage_summary.json"
    _write_csv(detail_path, detail_rows)
    _write_csv(summary_path, summary_rows)
    write_json(
        json_path,
        {
            "retrieval_path": str(args.retrieval_path),
            "embeddings_dir": str(args.embeddings_dir),
            "qa_json_path": str(args.qa_json_path),
            "conditions": args.conditions,
            "top_k_values": args.top_k_values,
            "summary": summary_rows,
            "detail_path": str(detail_path),
            "summary_path": str(summary_path),
        },
    )
    print(json.dumps({"summary": summary_rows, "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
