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
    "actual_skip_hard": "skip_chunks_actual_hard.npz",
    "predicted_trt_gaze": "gaze_chunks_predicted_trt.npz",
}
HYBRID_PREFIX = "hybrid_gaze_alpha_"
RERANK_PREFIX = "text_top"
RERANK_SUFFIX = "_gaze_rerank"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITION_FILES))
    parser.add_argument("--hybrid-alphas", nargs="*", type=float, default=[])
    parser.add_argument("--rerank-candidate-top-n", nargs="*", type=int, default=[])
    return parser.parse_args()


def _alpha_label(alpha: float) -> str:
    return f"{alpha:.6g}".replace(".", "p")


def _hybrid_condition(alpha: float) -> str:
    return f"{HYBRID_PREFIX}{_alpha_label(alpha)}"


def _alpha_from_hybrid_condition(condition: str) -> float:
    if not condition.startswith(HYBRID_PREFIX):
        raise ValueError(f"Not a hybrid condition: {condition}")
    return float(condition.removeprefix(HYBRID_PREFIX).replace("p", "."))


def _is_hybrid_condition(condition: str) -> bool:
    return condition.startswith(HYBRID_PREFIX)


def _rerank_condition(candidate_top_n: int) -> str:
    return f"{RERANK_PREFIX}{candidate_top_n}{RERANK_SUFFIX}"


def _candidate_top_n_from_rerank_condition(condition: str) -> int:
    if not _is_rerank_condition(condition):
        raise ValueError(f"Not a rerank condition: {condition}")
    value = condition.removeprefix(RERANK_PREFIX).removesuffix(RERANK_SUFFIX)
    return int(value)


def _is_rerank_condition(condition: str) -> bool:
    return condition.startswith(RERANK_PREFIX) and condition.endswith(RERANK_SUFFIX)


def _validate_alpha(alpha: float) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("hybrid alpha must satisfy 0 <= alpha <= 1.")


def _validate_candidate_top_n(candidate_top_n: int, top_k: int) -> None:
    if candidate_top_n <= 0:
        raise ValueError("rerank candidate_top_n must be positive.")
    if candidate_top_n < top_k:
        raise ValueError("rerank candidate_top_n must be greater than or equal to top_k.")


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


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or not np.isfinite(norm):
        return np.zeros_like(vector, dtype=np.float64)
    return vector / norm


def _records_by_chunk(
    rows: list[tuple[str, np.ndarray, dict[str, Any]]],
) -> dict[str, tuple[np.ndarray, dict[str, Any]]]:
    by_chunk: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
    for _, vector, record in rows:
        chunk_id = str(record["chunk_id"])
        by_chunk[chunk_id] = (np.asarray(vector, dtype=np.float64), record)
    return by_chunk


def _evidence_item(
    record: dict[str, Any], score: float, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    item = {
        "chunk_id": record["chunk_id"],
        "text": record["text"],
        "score": score,
        "char_start": record.get("char_start"),
        "char_end": record.get("char_end"),
    }
    if extra:
        item.update(extra)
    return item


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
        _evidence_item(record_by_id[embedding_id], score)
        for embedding_id, score in ranked_embedding_ids
    ]
    return RetrievalResult(
        example_id=example_id,
        reader_id=reader_id,
        condition=condition,
        ranked_chunks=ranked_chunks,
        metadata={"evidence": evidence},
    )


def _retrieve_hybrid_group(
    example_id: str,
    reader_id: str,
    alpha: float,
    text_rows: list[tuple[str, np.ndarray, dict[str, Any]]],
    gaze_rows: list[tuple[str, np.ndarray, dict[str, Any]]],
    query: np.ndarray,
    top_k: int,
) -> RetrievalResult:
    _validate_alpha(alpha)
    query_vector = _normalize_vector(np.asarray(query, dtype=np.float64))
    text_by_chunk = _records_by_chunk(text_rows)
    gaze_by_chunk = _records_by_chunk(gaze_rows)
    shared_chunk_ids = sorted(set(text_by_chunk) & set(gaze_by_chunk))
    if not shared_chunk_ids:
        raise ValueError(f"No shared chunks for hybrid retrieval on {example_id!r}/{reader_id!r}.")

    scored: list[tuple[str, float, float, float, dict[str, Any]]] = []
    for chunk_id in shared_chunk_ids:
        text_vector, text_record = text_by_chunk[chunk_id]
        gaze_vector, gaze_record = gaze_by_chunk[chunk_id]
        text_score = float(_normalize_vector(text_vector) @ query_vector)
        gaze_score = float(_normalize_vector(gaze_vector) @ query_vector)
        hybrid_score = alpha * text_score + (1.0 - alpha) * gaze_score
        record = dict(gaze_record)
        record["text"] = text_record.get("text", gaze_record.get("text", ""))
        record["char_start"] = text_record.get("char_start", gaze_record.get("char_start"))
        record["char_end"] = text_record.get("char_end", gaze_record.get("char_end"))
        scored.append((chunk_id, hybrid_score, text_score, gaze_score, record))

    scored.sort(key=lambda item: (-item[1], item[0]))
    selected = scored[: min(top_k, len(scored))]
    ranked_chunks = [(chunk_id, score) for chunk_id, score, _, _, _ in selected]
    evidence = [
        _evidence_item(
            record,
            hybrid_score,
            {
                "text_score": text_score,
                "gaze_score": gaze_score,
                "alpha": alpha,
            },
        )
        for chunk_id, hybrid_score, text_score, gaze_score, record in selected
    ]
    condition = _hybrid_condition(alpha)
    return RetrievalResult(
        example_id=example_id,
        reader_id=reader_id,
        condition=condition,
        ranked_chunks=ranked_chunks,
        metadata={
            "evidence": evidence,
            "alpha": alpha,
            "score_formula": "alpha*text+(1-alpha)*gaze",
        },
    )


def _retrieve_text_candidate_gaze_rerank_group(
    example_id: str,
    reader_id: str,
    candidate_top_n: int,
    text_rows: list[tuple[str, np.ndarray, dict[str, Any]]],
    gaze_rows: list[tuple[str, np.ndarray, dict[str, Any]]],
    query: np.ndarray,
    top_k: int,
) -> RetrievalResult:
    _validate_candidate_top_n(candidate_top_n, top_k)
    query_vector = _normalize_vector(np.asarray(query, dtype=np.float64))
    text_by_chunk = _records_by_chunk(text_rows)
    gaze_by_chunk = _records_by_chunk(gaze_rows)
    shared_chunk_ids = sorted(set(text_by_chunk) & set(gaze_by_chunk))
    if not shared_chunk_ids:
        raise ValueError(f"No shared chunks for rerank retrieval on {example_id!r}/{reader_id!r}.")

    scored: list[tuple[str, float, float, dict[str, Any]]] = []
    for chunk_id in shared_chunk_ids:
        text_vector, text_record = text_by_chunk[chunk_id]
        gaze_vector, gaze_record = gaze_by_chunk[chunk_id]
        text_score = float(_normalize_vector(text_vector) @ query_vector)
        gaze_score = float(_normalize_vector(gaze_vector) @ query_vector)
        record = dict(gaze_record)
        record["text"] = text_record.get("text", gaze_record.get("text", ""))
        record["char_start"] = text_record.get("char_start", gaze_record.get("char_start"))
        record["char_end"] = text_record.get("char_end", gaze_record.get("char_end"))
        scored.append((chunk_id, text_score, gaze_score, record))

    text_candidates = sorted(scored, key=lambda item: (-item[1], item[0]))[
        : min(candidate_top_n, len(scored))
    ]
    text_rank_by_chunk = {
        chunk_id: rank + 1 for rank, (chunk_id, _, _, _) in enumerate(text_candidates)
    }
    reranked = sorted(text_candidates, key=lambda item: (-item[2], -item[1], item[0]))
    selected = reranked[: min(top_k, len(reranked))]
    ranked_chunks = [(chunk_id, gaze_score) for chunk_id, _, gaze_score, _ in selected]
    evidence = [
        _evidence_item(
            record,
            gaze_score,
            {
                "text_score": text_score,
                "gaze_score": gaze_score,
                "text_candidate_rank": text_rank_by_chunk[chunk_id],
                "candidate_top_n": candidate_top_n,
            },
        )
        for chunk_id, text_score, gaze_score, record in selected
    ]
    condition = _rerank_condition(candidate_top_n)
    return RetrievalResult(
        example_id=example_id,
        reader_id=reader_id,
        condition=condition,
        ranked_chunks=ranked_chunks,
        metadata={
            "evidence": evidence,
            "candidate_top_n": candidate_top_n,
            "score_formula": "text top-N candidates, reranked by gaze score",
        },
    )


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("top_k must be positive.")
    hybrid_from_alphas = [_hybrid_condition(alpha) for alpha in args.hybrid_alphas]
    for alpha in args.hybrid_alphas:
        _validate_alpha(alpha)
    rerank_from_candidates = [
        _rerank_condition(candidate_top_n)
        for candidate_top_n in args.rerank_candidate_top_n
    ]
    for candidate_top_n in args.rerank_candidate_top_n:
        _validate_candidate_top_n(candidate_top_n, args.top_k)
    requested_conditions = list(
        dict.fromkeys([*args.conditions, *hybrid_from_alphas, *rerank_from_candidates])
    )
    hybrid_conditions = [condition for condition in requested_conditions if _is_hybrid_condition(condition)]
    for condition in hybrid_conditions:
        _validate_alpha(_alpha_from_hybrid_condition(condition))
    rerank_conditions = [condition for condition in requested_conditions if _is_rerank_condition(condition)]
    for condition in rerank_conditions:
        _validate_candidate_top_n(_candidate_top_n_from_rerank_condition(condition), args.top_k)

    unknown = {
        condition
        for condition in requested_conditions
        if (
            condition not in CONDITION_FILES
            and not _is_hybrid_condition(condition)
            and not _is_rerank_condition(condition)
        )
    }
    if unknown:
        raise ValueError(f"Unsupported conditions: {sorted(unknown)}")

    embeddings_dir = args.artifacts_dir / "embeddings"
    query_by_example, query_records = _load_query_npz(embeddings_dir / "query_embeddings.npz")
    retrievals: list[RetrievalResult] = []
    counts: dict[str, int] = {}
    grouped_by_condition: dict[
        str, dict[tuple[str, str | None], list[tuple[str, np.ndarray, dict[str, Any]]]]
    ] = {}
    required_embedding_conditions = {
        condition for condition in requested_conditions if condition in CONDITION_FILES
    }
    if hybrid_conditions or rerank_conditions:
        required_embedding_conditions.update({"text", "actual_gaze"})

    for condition in sorted(required_embedding_conditions):
        ids, embeddings, records = _load_embedding_npz(embeddings_dir / CONDITION_FILES[condition])
        grouped_by_condition[condition] = _group_records(ids, embeddings, records)

    for condition in requested_conditions:
        if _is_hybrid_condition(condition) or _is_rerank_condition(condition):
            continue
        grouped = grouped_by_condition[condition]
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

    for condition in hybrid_conditions:
        alpha = _alpha_from_hybrid_condition(condition)
        text_grouped = grouped_by_condition["text"]
        gaze_grouped = grouped_by_condition["actual_gaze"]
        hybrid_count = 0
        for (example_id, reader_id), gaze_rows in sorted(gaze_grouped.items()):
            if reader_id is None:
                continue
            if example_id not in query_by_example:
                raise KeyError(f"Missing query embedding for example {example_id!r}.")
            text_rows = text_grouped.get((example_id, None))
            if text_rows is None:
                raise KeyError(f"Missing text chunks for hybrid retrieval example {example_id!r}.")
            retrievals.append(
                _retrieve_hybrid_group(
                    example_id,
                    str(reader_id),
                    alpha,
                    text_rows,
                    gaze_rows,
                    query_by_example[example_id],
                    args.top_k,
                )
            )
            hybrid_count += 1
        counts[condition] = hybrid_count

    for condition in rerank_conditions:
        candidate_top_n = _candidate_top_n_from_rerank_condition(condition)
        text_grouped = grouped_by_condition["text"]
        gaze_grouped = grouped_by_condition["actual_gaze"]
        rerank_count = 0
        for (example_id, reader_id), gaze_rows in sorted(gaze_grouped.items()):
            if reader_id is None:
                continue
            if example_id not in query_by_example:
                raise KeyError(f"Missing query embedding for example {example_id!r}.")
            text_rows = text_grouped.get((example_id, None))
            if text_rows is None:
                raise KeyError(f"Missing text chunks for rerank retrieval example {example_id!r}.")
            retrievals.append(
                _retrieve_text_candidate_gaze_rerank_group(
                    example_id,
                    str(reader_id),
                    candidate_top_n,
                    text_rows,
                    gaze_rows,
                    query_by_example[example_id],
                    args.top_k,
                )
            )
            rerank_count += 1
        counts[condition] = rerank_count

    retrieval_dir = args.artifacts_dir / "retrieval"
    write_jsonl(retrieval_dir / "retrieval_results.jsonl", retrievals)
    summary = {
        "top_k": args.top_k,
        "conditions": requested_conditions,
        "hybrid_alphas": args.hybrid_alphas,
        "rerank_candidate_top_n": args.rerank_candidate_top_n,
        "query_examples": len(query_records),
        "retrieval_records": len(retrievals),
        "records_by_condition": counts,
        "retrieval_results_path": str(retrieval_dir / "retrieval_results.jsonl"),
    }
    write_json(retrieval_dir / "retrieval_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
