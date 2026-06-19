#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
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

from gaze_query_rag.evaluation.rag_rewardbench_compression import (
    COMPRESSION_CONDITIONS,
    FULL_CONTEXT,
    PREDICTED_ET_ONLY_COMPRESSION,
    QUERY_ONLY_COMPRESSION,
    QUERY_X_PREDICTED_ET_COMPRESSION,
    SUPPORTED_CONDITIONS,
    CompressedContext,
    combine_chunk_scores,
    compress_context_by_scores,
    predicted_trt_chunk_scores,
    split_context_chunks,
)
from gaze_query_rag.generation.llama import load_llama_generator
from gaze_query_rag.io.artifacts import ensure_parent, read_jsonl, to_jsonable, write_json
from gaze_query_rag.modeling.encoder import encode_passage_tokens, encode_query, load_e5_encoder
from gaze_query_rag.modeling.predicted_trt import (
    SKBOY_ET_REPO_ID,
    SKBOY_ET_WEIGHTS,
    load_trt_predictor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="jinzhuoran/RAG-RewardBench")
    parser.add_argument("--split", default="train")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts/rag_rewardbench")
    parser.add_argument("--generator-name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--subsets", nargs="*", default=None)
    parser.add_argument("--conditions", nargs="+", default=[FULL_CONTEXT])
    parser.add_argument("--context-budgets", nargs="+", type=int, default=[512])
    parser.add_argument("--compression-chunk-max-words", type=int, default=80)
    parser.add_argument("--compression-alpha", type=float, default=1.0)
    parser.add_argument("--compression-beta", type=float, default=0.25)
    parser.add_argument("--compression-tau-q", type=float, default=0.05)
    parser.add_argument("--compression-tau-g", type=float, default=0.1)
    parser.add_argument("--encoder-name", default="intfloat/e5-large-v2")
    parser.add_argument("--encoder-device", default=None)
    parser.add_argument("--predicted-trt-backend", choices=["skboy", "heuristic"], default="skboy")
    parser.add_argument("--predicted-trt-model-name", default=SKBOY_ET_REPO_ID)
    parser.add_argument("--predicted-trt-weights", default=SKBOY_ET_WEIGHTS)
    parser.add_argument("--predicted-trt-local-files-only", action="store_true")
    parser.add_argument("--score-normalization", choices=["sum", "mean"], default="mean")
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--store-text", action="store_true")
    return parser.parse_args()


def _row_id(index: int, row: dict[str, Any]) -> str:
    value = row.get("id")
    if value is not None and str(value):
        return str(value)
    return str(index)


def _format_prompt_prefix(row: dict[str, Any]) -> str:
    return _format_prompt_prefix_from_context(str(row["prompt"]), str(row["question"]))


def _format_prompt_prefix_from_context(context: str, question: str) -> str:
    references = context.strip()
    question = question.strip()
    return (
        "You are answering a retrieval-augmented question using only the provided references.\n\n"
        f"{references}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:\n"
    )


def _prediction_id(index: int, row: dict[str, Any], condition: str, budget: int | None) -> str:
    budget_label = "full" if budget is None else str(budget)
    return f"{_row_id(index, row)}::{condition}::budget={budget_label}"


def _prediction_key(row: dict[str, Any]) -> str:
    return str(row["id"])


def _existing_prediction_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {_prediction_key(row) for row in read_jsonl(path)}


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            json.dump(to_jsonable(row), fh, ensure_ascii=False)
            fh.write("\n")
        fh.flush()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_rows(args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    from datasets import load_dataset

    dataset = load_dataset(
        args.dataset_name,
        split=args.split,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )
    rows: list[tuple[int, dict[str, Any]]] = []
    allowed_subsets = set(args.subsets) if args.subsets else None
    for index, row in enumerate(dataset):
        item = dict(row)
        if allowed_subsets is not None and str(item.get("subset")) not in allowed_subsets:
            continue
        rows.append((index, item))
        if args.max_examples is not None and len(rows) >= args.max_examples:
            break
    return rows


def _condition_budget_pairs(args: argparse.Namespace) -> list[tuple[str, int | None]]:
    unknown = sorted(set(args.conditions) - SUPPORTED_CONDITIONS)
    if unknown:
        raise ValueError(f"Unsupported RAG-RewardBench conditions: {unknown}")
    if any(budget <= 0 for budget in args.context_budgets):
        raise ValueError("--context-budgets must contain positive integers.")
    pairs: list[tuple[str, int | None]] = []
    for condition in args.conditions:
        if condition == FULL_CONTEXT:
            pairs.append((condition, None))
        else:
            for budget in args.context_budgets:
                pairs.append((condition, int(budget)))
    return pairs


def _needs_query_encoder(conditions: list[str]) -> bool:
    return any(condition in {QUERY_ONLY_COMPRESSION, QUERY_X_PREDICTED_ET_COMPRESSION} for condition in conditions)


def _needs_trt_predictor(conditions: list[str]) -> bool:
    return any(
        condition in {PREDICTED_ET_ONLY_COMPRESSION, QUERY_X_PREDICTED_ET_COMPRESSION}
        for condition in conditions
    )


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0 or not np.isfinite(norm):
        return np.zeros_like(vector, dtype=np.float64)
    return vector / norm


def _chunk_query_scores(encoder, question: str, chunks, max_length: int) -> np.ndarray:
    query = encode_query(encoder, question)
    scores: list[float] = []
    for chunk in chunks:
        encoding = encode_passage_tokens(encoder, chunk.text, max_length=max_length)
        mask = encoding.attention_mask.astype(bool)
        hidden = encoding.hidden_states[mask]
        if hidden.size == 0:
            scores.append(0.0)
            continue
        chunk_vector = _normalize_vector(hidden.mean(axis=0))
        scores.append(float(np.dot(query, chunk_vector)))
    return np.asarray(scores, dtype=np.float64)


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _context_metadata(
    condition: str,
    budget: int | None,
    context: str,
    tokenizer,
    compressed: CompressedContext | None = None,
) -> dict[str, Any]:
    if compressed is None:
        token_count = _token_count(tokenizer, context)
        return {
            "condition": condition,
            "context_budget_tokens": budget,
            "original_context_tokens": token_count,
            "compressed_context_tokens": token_count,
            "compression_ratio": 1.0,
            "selected_chunk_count": None,
            "selected_chunks": [],
        }
    return {
        "condition": condition,
        "context_budget_tokens": budget,
        "original_context_tokens": compressed.original_token_count,
        "compressed_context_tokens": compressed.compressed_token_count,
        "compression_ratio": compressed.compression_ratio,
        "selected_chunk_count": len(compressed.selected_chunks),
        "selected_chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "index": chunk.index,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "score": score,
                "text": chunk.text,
            }
            for chunk, score in zip(compressed.selected_chunks, compressed.selected_scores)
        ],
    }


def _build_context_variant(
    row: dict[str, Any],
    condition: str,
    budget: int | None,
    args: argparse.Namespace,
    generator_tokenizer,
    encoder=None,
    trt_predictor=None,
) -> tuple[str, dict[str, Any]]:
    original_context = str(row["prompt"]).strip()
    if condition == FULL_CONTEXT:
        return original_context, _context_metadata(condition, budget, original_context, generator_tokenizer)
    if condition not in COMPRESSION_CONDITIONS:
        raise ValueError(f"Unsupported condition: {condition}")
    if budget is None:
        raise ValueError(f"Compressed condition {condition!r} requires a context budget.")

    chunks = split_context_chunks(original_context, max_words=args.compression_chunk_max_words)
    query_scores = None
    trt_scores = None
    if condition in {QUERY_ONLY_COMPRESSION, QUERY_X_PREDICTED_ET_COMPRESSION}:
        if encoder is None:
            raise ValueError(f"Condition {condition!r} requires an encoder.")
        query_scores = _chunk_query_scores(encoder, str(row["question"]), chunks, max_length=256)
    if condition in {PREDICTED_ET_ONLY_COMPRESSION, QUERY_X_PREDICTED_ET_COMPRESSION}:
        if trt_predictor is None:
            raise ValueError(f"Condition {condition!r} requires a TRT predictor.")
        trt_scores = predicted_trt_chunk_scores(trt_predictor, chunks)

    scores = combine_chunk_scores(
        condition,
        query_scores=query_scores,
        trt_scores=trt_scores,
        alpha=args.compression_alpha,
        beta=args.compression_beta,
        tau_q=args.compression_tau_q,
        tau_g=args.compression_tau_g,
    )
    compressed = compress_context_by_scores(
        original_context,
        chunks,
        scores,
        budget_tokens=budget,
        token_count_fn=lambda text: _token_count(generator_tokenizer, text),
    )
    return compressed.text, _context_metadata(condition, budget, original_context, generator_tokenizer, compressed)


def _score_answer_options_with_budget(
    generator,
    prompt_prefix: str,
    choices: list[str],
    max_length: int,
    score_normalization: str,
) -> np.ndarray:
    import torch

    if not choices:
        raise ValueError("choices must not be empty.")
    if max_length <= 0:
        raise ValueError("--max-length must be positive.")
    tokenizer = generator.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prefix_ids = tokenizer(prompt_prefix, add_special_tokens=True)["input_ids"]
    choice_ids = [
        tokenizer(choice, add_special_tokens=False)["input_ids"]
        for choice in choices
    ]
    max_choice_len = max(len(ids) for ids in choice_ids)
    if max_choice_len >= max_length:
        raise ValueError("A candidate response is longer than --max-length.")
    max_prefix_len = max_length - max_choice_len
    if len(prefix_ids) > max_prefix_len:
        prefix_ids = prefix_ids[-max_prefix_len:]

    rows = [prefix_ids + ids for ids in choice_ids]
    row_lengths = [len(row) for row in rows]
    padded = [
        row + [tokenizer.pad_token_id] * (max(row_lengths) - len(row))
        for row in rows
    ]
    attention = [
        [1] * len(row) + [0] * (max(row_lengths) - len(row))
        for row in rows
    ]
    input_ids = torch.tensor(padded, dtype=torch.long, device=generator.device)
    attention_mask = torch.tensor(attention, dtype=torch.long, device=generator.device)
    with torch.inference_mode():
        logits = generator.model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]
    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    valid_targets = torch.zeros_like(token_log_probs)
    start = max(len(prefix_ids) - 1, 0)
    for row_index, ids in enumerate(choice_ids):
        valid_targets[row_index, start : start + len(ids)] = 1.0
    scores = (token_log_probs * valid_targets).sum(dim=1)
    if score_normalization == "mean":
        scores = scores / valid_targets.sum(dim=1).clamp(min=1.0)
    elif score_normalization != "sum":
        raise ValueError(f"Unsupported score normalization: {score_normalization}")
    return scores.float().detach().cpu().numpy().astype(np.float64)


def _score_row(
    generator,
    index: int,
    row: dict[str, Any],
    condition: str,
    budget: int | None,
    context: str,
    context_metadata: dict[str, Any],
    store_text: bool,
    max_length: int,
    score_normalization: str,
) -> dict[str, Any]:
    choices = [str(row["chosen"]), str(row["reject"])]
    scores = _score_answer_options_with_budget(
        generator,
        _format_prompt_prefix_from_context(context, str(row["question"])),
        choices,
        max_length,
        score_normalization,
    )
    chosen_score = float(scores[0])
    reject_score = float(scores[1])
    correct = bool(chosen_score > reject_score)
    tie = bool(np.isclose(chosen_score, reject_score))
    output = {
        "id": _prediction_id(index, row, condition, budget),
        "row_id": _row_id(index, row),
        "row_index": index,
        "condition": condition,
        "context_budget_tokens": budget,
        "subset": str(row.get("subset", "")),
        "chosen_model": None if row.get("chosen_model") is None else str(row.get("chosen_model")),
        "reject_model": None if row.get("reject_model") is None else str(row.get("reject_model")),
        "chosen_score": chosen_score,
        "reject_score": reject_score,
        "score_delta": chosen_score - reject_score,
        "correct": correct,
        "tie": tie,
        "score_normalization": score_normalization,
        **context_metadata,
    }
    if store_text:
        output.update(
            {
                "prompt": str(row["prompt"]),
                "question": str(row["question"]),
                "compressed_context": context,
                "chosen": choices[0],
                "reject": choices[1],
            }
        )
    return output


def _summarize_predictions(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not rows:
        raise ValueError("No predictions to summarize.")
    correct = [1.0 if row["correct"] else 0.0 for row in rows]
    deltas = [float(row["score_delta"]) for row in rows]
    summary = {
        "prediction_records": len(rows),
        "accuracy": float(np.mean(correct)),
        "tie_rate": float(np.mean([1.0 if row["tie"] else 0.0 for row in rows])),
        "mean_score_delta": float(np.mean(deltas)),
        "median_score_delta": float(np.median(deltas)),
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = json.dumps(
            {
                "condition": str(row.get("condition", "")),
                "context_budget_tokens": row.get("context_budget_tokens"),
                "subset": str(row.get("subset", "")),
            },
            sort_keys=True,
        )
        grouped[key].append(row)
    by_subset = []
    for key, subset_rows in sorted(grouped.items()):
        group = json.loads(key)
        subset_correct = [1.0 if row["correct"] else 0.0 for row in subset_rows]
        subset_deltas = [float(row["score_delta"]) for row in subset_rows]
        by_subset.append(
            {
                "condition": group["condition"],
                "context_budget_tokens": group["context_budget_tokens"],
                "subset": group["subset"],
                "count": len(subset_rows),
                "accuracy": float(np.mean(subset_correct)),
                "tie_rate": float(np.mean([1.0 if row["tie"] else 0.0 for row in subset_rows])),
                "mean_score_delta": float(np.mean(subset_deltas)),
                "median_score_delta": float(np.median(subset_deltas)),
                "mean_compression_ratio": float(
                    np.mean([float(row.get("compression_ratio", 1.0)) for row in subset_rows])
                ),
                "mean_compressed_context_tokens": float(
                    np.mean([float(row.get("compressed_context_tokens", 0.0)) for row in subset_rows])
                ),
            }
        )
    by_condition: list[dict[str, Any]] = []
    condition_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = json.dumps(
            {
                "condition": str(row.get("condition", "")),
                "context_budget_tokens": row.get("context_budget_tokens"),
            },
            sort_keys=True,
        )
        condition_groups[key].append(row)
    for key, condition_rows in sorted(condition_groups.items()):
        group = json.loads(key)
        condition_correct = [1.0 if row["correct"] else 0.0 for row in condition_rows]
        condition_deltas = [float(row["score_delta"]) for row in condition_rows]
        by_condition.append(
            {
                "condition": group["condition"],
                "context_budget_tokens": group["context_budget_tokens"],
                "count": len(condition_rows),
                "accuracy": float(np.mean(condition_correct)),
                "tie_rate": float(np.mean([1.0 if row["tie"] else 0.0 for row in condition_rows])),
                "mean_score_delta": float(np.mean(condition_deltas)),
                "median_score_delta": float(np.median(condition_deltas)),
                "mean_compression_ratio": float(
                    np.mean([float(row.get("compression_ratio", 1.0)) for row in condition_rows])
                ),
                "mean_compressed_context_tokens": float(
                    np.mean([float(row.get("compressed_context_tokens", 0.0)) for row in condition_rows])
                ),
            }
        )
    summary["by_condition"] = by_condition
    return summary, by_subset


def _progress_iter(items: list[Any], args: argparse.Namespace):
    if args.disable_progress:
        return items
    from tqdm import tqdm

    return tqdm(items, total=len(items), desc="rag_rewardbench", unit="row")


def main() -> None:
    args = parse_args()
    if args.flush_every <= 0:
        raise ValueError("--flush-every must be positive.")
    condition_budget_pairs = _condition_budget_pairs(args)

    prediction_path = args.artifacts_dir / "predictions" / "rag_rewardbench_predictions.jsonl"
    if args.overwrite and prediction_path.exists():
        prediction_path.unlink()

    rows = _load_rows(args)
    if not rows:
        raise ValueError("No RAG-RewardBench rows matched the requested filters.")

    existing = _existing_prediction_keys(prediction_path) if args.resume else set()
    pending: list[tuple[int, dict[str, Any], list[tuple[str, int | None]]]] = []
    requested_records = 0
    for index, row in rows:
        row_pairs = [
            (condition, budget)
            for condition, budget in condition_budget_pairs
            if _prediction_id(index, row, condition, budget) not in existing
        ]
        requested_records += len(condition_budget_pairs)
        if row_pairs:
            pending.append((index, row, row_pairs))

    generator = load_llama_generator(args.generator_name, args.device, args.cache_dir, args.dtype)
    encoder = None
    if _needs_query_encoder(args.conditions):
        encoder = load_e5_encoder(
            args.encoder_name,
            args.encoder_device or args.device,
            args.cache_dir,
        )
    trt_predictor = None
    if _needs_trt_predictor(args.conditions):
        trt_predictor = load_trt_predictor(
            backend=args.predicted_trt_backend,
            repo_id=args.predicted_trt_model_name,
            weights_filename=args.predicted_trt_weights,
            cache_dir=args.cache_dir,
            local_files_only=args.predicted_trt_local_files_only,
        )

    buffer: list[dict[str, Any]] = []
    newly_written = 0
    for index, row, row_pairs in _progress_iter(pending, args):
        for condition, budget in row_pairs:
            context, context_metadata = _build_context_variant(
                row,
                condition,
                budget,
                args,
                generator.tokenizer,
                encoder=encoder,
                trt_predictor=trt_predictor,
            )
            buffer.append(
                _score_row(
                    generator,
                    index,
                    row,
                    condition,
                    budget,
                    context,
                    context_metadata,
                    args.store_text,
                    args.max_length,
                    args.score_normalization,
                )
            )
        if len(buffer) >= args.flush_every:
            _append_jsonl(prediction_path, buffer)
            newly_written += len(buffer)
            buffer = []
    if buffer:
        _append_jsonl(prediction_path, buffer)
        newly_written += len(buffer)

    prediction_rows = read_jsonl(prediction_path)
    summary, by_subset = _summarize_predictions(prediction_rows)
    summary.update(
        {
            "dataset_name": args.dataset_name,
            "split": args.split,
            "generator_name": args.generator_name,
            "device": generator.device,
            "dtype": args.dtype,
            "max_length": args.max_length,
            "score_normalization": args.score_normalization,
            "conditions": args.conditions,
            "context_budgets": args.context_budgets,
            "compression_alpha": args.compression_alpha,
            "compression_beta": args.compression_beta,
            "compression_tau_q": args.compression_tau_q,
            "compression_tau_g": args.compression_tau_g,
            "encoder_name": args.encoder_name if _needs_query_encoder(args.conditions) else None,
            "predicted_trt_backend": args.predicted_trt_backend if _needs_trt_predictor(args.conditions) else None,
            "predicted_trt_model_name": (
                args.predicted_trt_model_name if _needs_trt_predictor(args.conditions) else None
            ),
            "requested_records": requested_records,
            "skipped_existing": requested_records - newly_written,
            "newly_written": newly_written,
            "predictions_path": str(prediction_path),
            "by_subset_path": str(args.artifacts_dir / "results" / "rag_rewardbench_by_subset.csv"),
            "by_condition_path": str(args.artifacts_dir / "results" / "rag_rewardbench_by_condition.csv"),
        }
    )
    write_json(args.artifacts_dir / "results" / "rag_rewardbench_summary.json", summary)
    _write_csv(args.artifacts_dir / "results" / "rag_rewardbench_by_subset.csv", by_subset)
    _write_csv(args.artifacts_dir / "results" / "rag_rewardbench_by_condition.csv", summary["by_condition"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
