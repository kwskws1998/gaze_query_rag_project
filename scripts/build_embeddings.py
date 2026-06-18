#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gaze_query_rag.data.chunking import chunk_paragraph
from gaze_query_rag.io.artifacts import load_aligned_examples, write_json
from gaze_query_rag.modeling.encoder import (
    build_word_to_token_alignment,
    encode_passage_tokens,
    encode_query,
    load_e5_encoder,
)
from gaze_query_rag.modeling.gaze_features import (
    compute_gaze_distribution,
    shuffle_reader_gaze,
    word_trt_to_token_trt,
)
from gaze_query_rag.modeling.gaze_query_attention import (
    build_gaze_view_chunk_embeddings,
    build_text_chunk_embeddings,
)
from gaze_query_rag.modeling.mean_gaze import MEAN_GAZE_CONDITION, build_mean_gaze_chunk_embeddings
from gaze_query_rag.modeling.predicted_trt import (
    PREDICTED_TRT_GAZE_CONDITION,
    SKBOY_ET_REPO_ID,
    SKBOY_ET_WEIGHTS,
    build_predicted_trt_chunk_embeddings,
    load_trt_predictor,
    predicted_words_to_token_trt,
)
from gaze_query_rag.modeling.skip_attention import (
    SKIP_HARD_CONDITION,
    build_skip_hard_chunk_embeddings,
    word_skip_to_token_query_weights,
)
from gaze_query_rag.schemas import AlignedExample, Chunk, TokenEncoding


CONDITION_CHOICES = (
    "text",
    "actual_gaze",
    MEAN_GAZE_CONDITION,
    "shuffled_gaze",
    SKIP_HARD_CONDITION,
    PREDICTED_TRT_GAZE_CONDITION,
)
DEFAULT_CONDITIONS = (
    "text",
    "actual_gaze",
    MEAN_GAZE_CONDITION,
    "shuffled_gaze",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-path", type=Path, default=ROOT / "artifacts/data/aligned_examples.jsonl")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--encoder-backend", choices=["e5", "hash"], default="e5")
    parser.add_argument("--encoder-name", default="intfloat/e5-large-v2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--query-max-length", type=int, default=128)
    parser.add_argument("--hash-dim", type=int, default=128)
    parser.add_argument("--chunk-strategy", choices=["sentence", "fixed_words", "whole_paragraph"], default="sentence")
    parser.add_argument("--max-words", type=int, default=80)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--conditions", nargs="+", choices=CONDITION_CHOICES, default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--predicted-trt-backend", choices=["skboy", "heuristic"], default="skboy")
    parser.add_argument("--predicted-trt-model-name", default=SKBOY_ET_REPO_ID)
    parser.add_argument("--predicted-trt-weights", default=SKBOY_ET_WEIGHTS)
    parser.add_argument("--predicted-trt-local-files-only", action="store_true")
    return parser.parse_args()


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or not np.isfinite(norm):
        return np.zeros_like(vector, dtype=np.float64)
    return vector / norm


def _hash_vector(text: str, dim: int) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = np.empty(dim, dtype=np.float64)
    for index in range(dim):
        byte = digest[index % len(digest)]
        values[index] = (byte / 255.0) * 2.0 - 1.0
    return _normalize(values)


def _word_spans(text: str) -> list[tuple[int, int, str]]:
    return [(match.start(), match.end(), match.group(0)) for match in re.finditer(r"\S+", text)]


def _hash_passage_encoding(text: str, dim: int) -> TokenEncoding:
    spans = _word_spans(text)
    if not spans:
        raise ValueError("Cannot encode an empty paragraph.")
    offsets = [(start, end) for start, end, _ in spans]
    tokens = [token for _, _, token in spans]
    hidden = np.stack([_hash_vector(f"passage::{token.lower()}", dim) for token in tokens], axis=0)
    return TokenEncoding(
        input_ids=np.arange(len(tokens), dtype=np.int64),
        attention_mask=np.ones(len(tokens), dtype=np.int64),
        tokens=tokens,
        offset_mapping=offsets,
        hidden_states=hidden,
    )


def _hash_query(question: str, dim: int) -> np.ndarray:
    tokens = re.findall(r"\S+", question)
    if not tokens:
        raise ValueError("Cannot encode an empty query.")
    pooled = np.mean([_hash_vector(f"query::{token.lower()}", dim) for token in tokens], axis=0)
    return _normalize(pooled)


def _chunks_with_token_indices(chunks: list[Chunk], token_offsets: list[tuple[int, int]]) -> list[Chunk]:
    out: list[Chunk] = []
    for chunk in chunks:
        token_indices = [
            index
            for index, (start, end) in enumerate(token_offsets)
            if start >= 0 and end > start and end > chunk.char_start and start < chunk.char_end
        ]
        if not token_indices:
            raise ValueError(f"Chunk {chunk.chunk_id!r} has no overlapping token offsets.")
        out.append(
            Chunk(
                chunk_id=chunk.chunk_id,
                paragraph_id=chunk.paragraph_id,
                text=chunk.text,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                token_indices=token_indices,
            )
        )
    return out


def _records_for_embeddings(
    example: AlignedExample,
    reader_id: str | None,
    condition: str,
    chunks: list[Chunk],
    embeddings: dict[str, np.ndarray],
) -> tuple[list[str], list[np.ndarray], list[dict[str, Any]]]:
    ids: list[str] = []
    vectors: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for chunk in chunks:
        embedding_id = (
            f"{example.qa.example_id}::{condition}::{reader_id or 'none'}::{chunk.chunk_id}"
        )
        ids.append(embedding_id)
        vectors.append(embeddings[chunk.chunk_id].astype(np.float32))
        records.append(
            {
                "embedding_id": embedding_id,
                "example_id": example.qa.example_id,
                "reader_id": reader_id,
                "condition": condition,
                "chunk_id": chunk.chunk_id,
                "paragraph_id": example.qa.paragraph_id,
                "text": chunk.text,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "token_indices": chunk.token_indices,
            }
        )
    return ids, vectors, records


def _save_embedding_npz(path: Path, ids: list[str], vectors: list[np.ndarray], records: list[dict[str, Any]]) -> None:
    if not vectors:
        raise ValueError(f"No embeddings to save for {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=np.asarray(ids, dtype=object),
        embeddings=np.stack(vectors, axis=0).astype(np.float32),
        records_json=np.asarray(json.dumps(records, ensure_ascii=False), dtype=object),
    )


def _save_query_npz(path: Path, query_rows: list[dict[str, Any]], query_vectors: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        example_ids=np.asarray([row["example_id"] for row in query_rows], dtype=object),
        queries=np.stack(query_vectors, axis=0).astype(np.float32),
        records_json=np.asarray(json.dumps(query_rows, ensure_ascii=False), dtype=object),
    )


def _reader_words(example: AlignedExample, reader_id: str) -> list[str]:
    return [record.word for record in example.reader_gaze[reader_id]]


def _reader_trt(example: AlignedExample, reader_id: str) -> np.ndarray:
    return np.asarray([record.trt for record in example.reader_gaze[reader_id]], dtype=np.float64)


def _reader_skip(example: AlignedExample, reader_id: str) -> np.ndarray:
    return np.asarray(
        [0.0 if record.skip is None else record.skip for record in example.reader_gaze[reader_id]],
        dtype=np.float64,
    )


def main() -> None:
    args = parse_args()
    selected_conditions = set(args.conditions)
    aligned_examples = load_aligned_examples(args.aligned_path)
    if args.max_examples is not None:
        aligned_examples = aligned_examples[: args.max_examples]
    if not aligned_examples:
        raise ValueError(f"No aligned examples found at {args.aligned_path}.")

    encoder = None
    if args.encoder_backend == "e5":
        encoder = load_e5_encoder(args.encoder_name, args.device, args.cache_dir)

    text_ids: list[str] = []
    text_vectors: list[np.ndarray] = []
    text_records: list[dict[str, Any]] = []
    actual_ids: list[str] = []
    actual_vectors: list[np.ndarray] = []
    actual_records: list[dict[str, Any]] = []
    mean_ids: list[str] = []
    mean_vectors: list[np.ndarray] = []
    mean_records: list[dict[str, Any]] = []
    shuffled_ids: list[str] = []
    shuffled_vectors: list[np.ndarray] = []
    shuffled_records: list[dict[str, Any]] = []
    skip_ids: list[str] = []
    skip_vectors: list[np.ndarray] = []
    skip_records: list[dict[str, Any]] = []
    predicted_ids: list[str] = []
    predicted_vectors: list[np.ndarray] = []
    predicted_records: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    query_vectors: list[np.ndarray] = []
    trt_predictor = None
    if PREDICTED_TRT_GAZE_CONDITION in selected_conditions:
        trt_predictor = load_trt_predictor(
            backend=args.predicted_trt_backend,
            repo_id=args.predicted_trt_model_name,
            weights_filename=args.predicted_trt_weights,
            cache_dir=args.cache_dir,
            local_files_only=args.predicted_trt_local_files_only,
        )

    for example_index, example in enumerate(aligned_examples):
        chunks = chunk_paragraph(
            example.qa.paragraph_id,
            example.qa.paragraph_text,
            strategy=args.chunk_strategy,
            max_words=args.max_words,
            stride=args.stride,
        )
        if args.encoder_backend == "e5":
            assert encoder is not None
            encoding = encode_passage_tokens(encoder, example.qa.paragraph_text, args.max_length)
            query_vector = encode_query(encoder, example.qa.question, args.query_max_length)
        else:
            encoding = _hash_passage_encoding(example.qa.paragraph_text, args.hash_dim)
            query_vector = _hash_query(example.qa.question, args.hash_dim)
        if encoding.offset_mapping is None:
            raise ValueError("Token offset mapping is required for gaze alignment.")
        chunks = _chunks_with_token_indices(chunks, encoding.offset_mapping)
        if "text" in selected_conditions:
            text_embeddings = build_text_chunk_embeddings(encoding.hidden_states, chunks)
            ids, vectors, records = _records_for_embeddings(example, None, "text", chunks, text_embeddings)
            text_ids.extend(ids)
            text_vectors.extend(vectors)
            text_records.extend(records)

        if PREDICTED_TRT_GAZE_CONDITION in selected_conditions:
            if trt_predictor is None:
                raise RuntimeError("TRT predictor was not loaded.")
            predicted_words = trt_predictor.predict_words(example.qa.paragraph_text)
            predicted_word_to_token = build_word_to_token_alignment(
                example.qa.paragraph_text,
                [row.word for row in predicted_words],
                encoding.offset_mapping,
            )
            predicted_token_trt = predicted_words_to_token_trt(
                predicted_words,
                predicted_word_to_token,
                encoding.hidden_states.shape[0],
            )
            predicted_embeddings = build_predicted_trt_chunk_embeddings(
                encoding.hidden_states,
                predicted_token_trt,
                chunks,
            )
            ids, vectors, records = _records_for_embeddings(
                example,
                None,
                PREDICTED_TRT_GAZE_CONDITION,
                chunks,
                predicted_embeddings,
            )
            predicted_ids.extend(ids)
            predicted_vectors.extend(vectors)
            predicted_records.extend(records)

        gaze_conditions = {"actual_gaze", MEAN_GAZE_CONDITION, "shuffled_gaze"}
        skip_conditions = {SKIP_HARD_CONDITION}
        gaze_by_reader: dict[str, np.ndarray] = {}
        skip_weights_by_reader: dict[str, np.ndarray] = {}
        if selected_conditions & (gaze_conditions | skip_conditions):
            for reader_id in sorted(example.reader_gaze):
                words = _reader_words(example, reader_id)
                word_to_token = build_word_to_token_alignment(
                    example.qa.paragraph_text, words, encoding.offset_mapping
                )
                if selected_conditions & gaze_conditions:
                    token_trt = word_trt_to_token_trt(
                        _reader_trt(example, reader_id),
                        word_to_token,
                        encoding.hidden_states.shape[0],
                    )
                    gaze_by_reader[reader_id] = compute_gaze_distribution(token_trt)
                if SKIP_HARD_CONDITION in selected_conditions:
                    skip_weights_by_reader[reader_id] = word_skip_to_token_query_weights(
                        _reader_skip(example, reader_id),
                        word_to_token,
                        encoding.hidden_states.shape[0],
                    )

        if MEAN_GAZE_CONDITION in selected_conditions:
            mean_embeddings = build_mean_gaze_chunk_embeddings(
                encoding.hidden_states, gaze_by_reader, chunks
            )
            ids, vectors, records = _records_for_embeddings(
                example, None, MEAN_GAZE_CONDITION, chunks, mean_embeddings
            )
            mean_ids.extend(ids)
            mean_vectors.extend(vectors)
            mean_records.extend(records)

        shuffled_gaze_by_reader: dict[str, np.ndarray] = {}
        if "shuffled_gaze" in selected_conditions:
            shuffled_gaze_by_reader = shuffle_reader_gaze(gaze_by_reader, seed=args.seed + example_index)
        for reader_id in sorted(gaze_by_reader):
            if "actual_gaze" in selected_conditions:
                actual_embeddings = build_gaze_view_chunk_embeddings(
                    encoding.hidden_states, gaze_by_reader[reader_id], chunks
                )
                ids, vectors, records = _records_for_embeddings(
                    example, reader_id, "actual_gaze", chunks, actual_embeddings
                )
                actual_ids.extend(ids)
                actual_vectors.extend(vectors)
                actual_records.extend(records)

            if "shuffled_gaze" in selected_conditions:
                shuffled_embeddings = build_gaze_view_chunk_embeddings(
                    encoding.hidden_states, shuffled_gaze_by_reader[reader_id], chunks
                )
                ids, vectors, records = _records_for_embeddings(
                    example, reader_id, "shuffled_gaze", chunks, shuffled_embeddings
                )
                shuffled_ids.extend(ids)
                shuffled_vectors.extend(vectors)
                shuffled_records.extend(records)

        for reader_id in sorted(skip_weights_by_reader):
            skip_embeddings = build_skip_hard_chunk_embeddings(
                encoding.hidden_states, skip_weights_by_reader[reader_id], chunks
            )
            ids, vectors, records = _records_for_embeddings(
                example, reader_id, SKIP_HARD_CONDITION, chunks, skip_embeddings
            )
            skip_ids.extend(ids)
            skip_vectors.extend(vectors)
            skip_records.extend(records)

        query_rows.append(
            {
                "example_id": example.qa.example_id,
                "paragraph_id": example.qa.paragraph_id,
                "question": example.qa.question,
                "choices": example.qa.choices,
                "answer_index": example.qa.answer_index,
            }
        )
        query_vectors.append(query_vector.astype(np.float32))

    embeddings_dir = args.artifacts_dir / "embeddings"
    if "text" in selected_conditions:
        _save_embedding_npz(embeddings_dir / "text_chunks.npz", text_ids, text_vectors, text_records)
    if "actual_gaze" in selected_conditions:
        _save_embedding_npz(
            embeddings_dir / "gaze_chunks_actual.npz", actual_ids, actual_vectors, actual_records
        )
    if MEAN_GAZE_CONDITION in selected_conditions:
        _save_embedding_npz(embeddings_dir / "gaze_chunks_mean.npz", mean_ids, mean_vectors, mean_records)
    if "shuffled_gaze" in selected_conditions:
        _save_embedding_npz(
            embeddings_dir / "gaze_chunks_shuffled.npz", shuffled_ids, shuffled_vectors, shuffled_records
        )
    if SKIP_HARD_CONDITION in selected_conditions:
        _save_embedding_npz(
            embeddings_dir / "skip_chunks_actual_hard.npz", skip_ids, skip_vectors, skip_records
        )
    if PREDICTED_TRT_GAZE_CONDITION in selected_conditions:
        _save_embedding_npz(
            embeddings_dir / "gaze_chunks_predicted_trt.npz",
            predicted_ids,
            predicted_vectors,
            predicted_records,
        )
    _save_query_npz(embeddings_dir / "query_embeddings.npz", query_rows, query_vectors)
    summary = {
        "encoder_backend": args.encoder_backend,
        "encoder_name": args.encoder_name if args.encoder_backend == "e5" else None,
        "device": encoder.device if encoder is not None else "hash",
        "conditions": args.conditions,
        "aligned_examples": len(aligned_examples),
        "text_embeddings": len(text_ids),
        "actual_gaze_embeddings": len(actual_ids),
        "mean_gaze_embeddings": len(mean_ids),
        "shuffled_gaze_embeddings": len(shuffled_ids),
        "actual_skip_hard_embeddings": len(skip_ids),
        "predicted_trt_gaze_embeddings": len(predicted_ids),
        "predicted_trt_backend": (
            args.predicted_trt_backend
            if PREDICTED_TRT_GAZE_CONDITION in selected_conditions
            else None
        ),
        "predicted_trt_model_name": (
            args.predicted_trt_model_name
            if PREDICTED_TRT_GAZE_CONDITION in selected_conditions
            else None
        ),
        "query_embeddings": len(query_rows),
    }
    write_json(embeddings_dir / "embedding_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
