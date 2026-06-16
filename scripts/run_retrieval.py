#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gaze_query_rag.io.artifacts import write_json, write_jsonl
from gaze_query_rag.retrieval.index import build_in_memory_index, search_index
from gaze_query_rag.schemas import RetrievalResult


CONDITION_FILES = {
    "text": "text_chunks.npz",
    "actual_gaze": "gaze_chunks_actual.npz",
    "mean_gaze": "gaze_chunks_mean.npz",
    "shuffled_gaze": "gaze_chunks_shuffled.npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITION_FILES))
    return parser.parse_args()


def _load_embedding_npz(path: Path) -> tuple[list[str], np.ndarray, list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    ids = [str(item) for item in data["ids"].tolist()]
    embeddings = np.asarray(data["embeddings"], dtype=np.float64)
    records = json.loads(str(data["records_json"].item()))
    if len(ids) != len(records) or len(ids) != embeddings.shape[0]:
        raise ValueError(f"Inconsistent embedding artifact lengths in {path}.")
    return ids, embeddings, records


def _load_query_npz(path: Path) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    example_ids = [str(item) for item in data["example_ids"].tolist()]
    queries = np.asarray(data["queries"], dtype=np.float64)
    records = json.loads(str(data["records_json"].item()))
    return (
        {example_id: queries[index] for index, example_id in enumerate(example_ids)},
        {record["example_id"]: record for record in records},
    )


def _group_records(
    ids: list[str], embeddings: np.ndarray, records: list[dict[str, Any]]
) -> dict[tuple[str, str | None], list[tuple[str, np.ndarray, dict[str, Any]]]]:
    grouped: dict[tuple[str, str | None], list[tuple[str, np.ndarray, dict[str, Any]]]] = defaultdict(list)
    for index, record in enumerate(records):
        key = (str(record["example_id"]), record.get("reader_id"))
        grouped[key].append((ids[index], embeddings[index], record))
    return grouped


def _retrieve_group(
    example_id: str,
    reader_id: str | None,
    condition: str,
    rows: list[tuple[str, np.ndarray, dict[str, Any]]],
    query: np.ndarray,
    top_k: int,
) -> RetrievalResult:
    chunk_embeddings = {embedding_id: vector for embedding_id, vector, _ in rows}
    record_by_id = {embedding_id: record for embedding_id, _, record in rows}
    ranked_embedding_ids = search_index(build_in_memory_index(chunk_embeddings), query, top_k)
    ranked_chunks = [
        (str(record_by_id[embedding_id]["chunk_id"]), score)
        for embedding_id, score in ranked_embedding_ids
    ]
    evidence = [
        {
            "chunk_id": record_by_id[embedding_id]["chunk_id"],
            "text": record_by_id[embedding_id]["text"],
            "score": score,
        }
        for embedding_id, score in ranked_embedding_ids
    ]
    return RetrievalResult(
        example_id=example_id,
        reader_id=reader_id,
        condition=condition,
        ranked_chunks=ranked_chunks,
        metadata={"evidence": evidence},
    )


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("top_k must be positive.")
    unknown = set(args.conditions) - set(CONDITION_FILES)
    if unknown:
        raise ValueError(f"Unsupported conditions: {sorted(unknown)}")

    embeddings_dir = args.artifacts_dir / "embeddings"
    query_by_example, query_records = _load_query_npz(embeddings_dir / "query_embeddings.npz")
    retrievals: list[RetrievalResult] = []
    counts: dict[str, int] = {}
    for condition in args.conditions:
        ids, embeddings, records = _load_embedding_npz(embeddings_dir / CONDITION_FILES[condition])
        grouped = _group_records(ids, embeddings, records)
        for (example_id, reader_id), rows in sorted(grouped.items()):
            if example_id not in query_by_example:
                raise KeyError(f"Missing query embedding for example {example_id!r}.")
            retrievals.append(
                _retrieve_group(
                    example_id,
                    reader_id,
                    condition,
                    rows,
                    query_by_example[example_id],
                    args.top_k,
                )
            )
        counts[condition] = len(grouped)

    retrieval_dir = args.artifacts_dir / "retrieval"
    write_jsonl(retrieval_dir / "retrieval_results.jsonl", retrievals)
    summary = {
        "top_k": args.top_k,
        "conditions": args.conditions,
        "query_examples": len(query_records),
        "retrieval_records": len(retrievals),
        "records_by_condition": counts,
        "retrieval_results_path": str(retrieval_dir / "retrieval_results.jsonl"),
    }
    write_json(retrieval_dir / "retrieval_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
