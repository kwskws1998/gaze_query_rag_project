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
from gaze_query_rag.data.hf_onestop_qa import load_local_onestop_qa_json, load_onestop_qa
from gaze_query_rag.io.artifacts import write_json
from gaze_query_rag.modeling.encoder import encode_passage_tokens, encode_query, load_e5_encoder
from gaze_query_rag.modeling.gaze_query_attention import build_text_chunk_embeddings
from gaze_query_rag.schemas import Chunk, QAExample, TokenEncoding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qa-json-path",
        type=Path,
        default=ROOT / "OneStop-Eye-Movements/data_preprocessing/onestop_qa.json",
    )
    parser.add_argument("--qa-dataset-name", default="malmaud/onestop_qa")
    parser.add_argument("--qa-split", default=None)
    parser.add_argument("--use-hf", action="store_true")
    parser.add_argument("--shuffle-choices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--choice-seed", type=int, default=13)
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
    parser.add_argument("--max-examples", type=int, default=None)
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
    example: QAExample,
    chunks: list[Chunk],
    embeddings: dict[str, np.ndarray],
) -> tuple[list[str], list[np.ndarray], list[dict[str, Any]]]:
    ids: list[str] = []
    vectors: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for chunk in chunks:
        embedding_id = f"{example.example_id}::text::none::{chunk.chunk_id}"
        ids.append(embedding_id)
        vectors.append(embeddings[chunk.chunk_id].astype(np.float32))
        records.append(
            {
                "embedding_id": embedding_id,
                "example_id": example.example_id,
                "reader_id": None,
                "condition": "text",
                "chunk_id": chunk.chunk_id,
                "paragraph_id": example.paragraph_id,
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


def _load_qa_examples(args: argparse.Namespace) -> list[QAExample]:
    if args.use_hf:
        examples = load_onestop_qa(
            args.qa_dataset_name,
            args.qa_split,
            cache_dir=None,
            shuffle_choices=args.shuffle_choices,
            choice_seed=args.choice_seed,
        )
    else:
        examples = load_local_onestop_qa_json(
            args.qa_json_path,
            shuffle_choices=args.shuffle_choices,
            choice_seed=args.choice_seed,
        )
    if args.max_examples is not None:
        if args.max_examples <= 0:
            raise ValueError("max_examples must be positive when provided.")
        examples = examples[: args.max_examples]
    if not examples:
        raise ValueError("No QA examples loaded.")
    return examples


def main() -> None:
    args = parse_args()
    qa_examples = _load_qa_examples(args)
    encoder = None
    if args.encoder_backend == "e5":
        encoder = load_e5_encoder(args.encoder_name, args.device, args.cache_dir)

    text_ids: list[str] = []
    text_vectors: list[np.ndarray] = []
    text_records: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    query_vectors: list[np.ndarray] = []

    for example in qa_examples:
        chunks = chunk_paragraph(
            example.paragraph_id,
            example.paragraph_text,
            strategy=args.chunk_strategy,
            max_words=args.max_words,
            stride=args.stride,
        )
        if args.encoder_backend == "e5":
            assert encoder is not None
            encoding = encode_passage_tokens(encoder, example.paragraph_text, args.max_length)
            query_vector = encode_query(encoder, example.question, args.query_max_length)
        else:
            encoding = _hash_passage_encoding(example.paragraph_text, args.hash_dim)
            query_vector = _hash_query(example.question, args.hash_dim)
        if encoding.offset_mapping is None:
            raise ValueError("Token offset mapping is required for chunk alignment.")
        chunks = _chunks_with_token_indices(chunks, encoding.offset_mapping)
        text_embeddings = build_text_chunk_embeddings(encoding.hidden_states, chunks)
        ids, vectors, records = _records_for_embeddings(example, chunks, text_embeddings)
        text_ids.extend(ids)
        text_vectors.extend(vectors)
        text_records.extend(records)
        query_rows.append(
            {
                "example_id": example.example_id,
                "paragraph_id": example.paragraph_id,
                "question": example.question,
                "choices": example.choices,
                "answer_index": example.answer_index,
                "choice_shuffle": example.metadata.get("choice_shuffle"),
            }
        )
        query_vectors.append(query_vector.astype(np.float32))

    embeddings_dir = args.artifacts_dir / "embeddings"
    _save_embedding_npz(embeddings_dir / "text_chunks.npz", text_ids, text_vectors, text_records)
    _save_query_npz(embeddings_dir / "query_embeddings.npz", query_rows, query_vectors)
    summary = {
        "encoder_backend": args.encoder_backend,
        "encoder_name": args.encoder_name if args.encoder_backend == "e5" else None,
        "device": encoder.device if encoder is not None else "hash",
        "qa_examples": len(qa_examples),
        "shuffle_choices": args.shuffle_choices,
        "choice_seed": args.choice_seed if args.shuffle_choices else None,
        "text_embeddings": len(text_ids),
        "query_embeddings": len(query_rows),
        "conditions": ["text"],
    }
    write_json(embeddings_dir / "embedding_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
