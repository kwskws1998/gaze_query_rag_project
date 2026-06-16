#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gaze_query_rag.generation.answer_parser import parse_choice
from gaze_query_rag.generation.llama import generate_answer, load_llama_generator, score_answer_options
from gaze_query_rag.generation.prompt import format_bare_mcqa_prompt, format_mcqa_prompt
from gaze_query_rag.data.hf_onestop_qa import load_local_onestop_qa_json, load_onestop_qa
from gaze_query_rag.io.artifacts import (
    ensure_parent,
    load_aligned_examples,
    read_jsonl,
    to_jsonable,
    write_json,
)
from gaze_query_rag.schemas import Chunk, PredictionRecord, RetrievalResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-path", type=Path, default=ROOT / "artifacts/data/aligned_examples.jsonl")
    parser.add_argument(
        "--qa-json-path",
        type=Path,
        default=None,
    )
    parser.add_argument("--qa-dataset-name", default="malmaud/onestop_qa")
    parser.add_argument("--qa-split", default=None)
    parser.add_argument("--use-hf", action="store_true")
    parser.add_argument("--shuffle-choices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--choice-seed", type=int, default=13)
    parser.add_argument("--retrieval-path", type=Path, default=ROOT / "artifacts/retrieval/retrieval_results.jsonl")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--generator-name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--scoring-mode", choices=["generate", "loglik"], default="generate")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--include-bare-model", action="store_true")
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    return parser.parse_args()


def _load_qa_by_id(args: argparse.Namespace) -> dict[str, dict]:
    qa_by_id = {}
    if args.qa_json_path is not None or args.use_hf:
        examples = (
            load_onestop_qa(
                args.qa_dataset_name,
                args.qa_split,
                cache_dir=None,
                shuffle_choices=args.shuffle_choices,
                choice_seed=args.choice_seed,
            )
            if args.use_hf
            else load_local_onestop_qa_json(
                args.qa_json_path,
                shuffle_choices=args.shuffle_choices,
                choice_seed=args.choice_seed,
            )
        )
        for item in examples:
            qa_by_id[item.example_id] = {
                "question": item.question,
                "choices": item.choices,
                "answer_index": item.answer_index,
                "choice_shuffle": item.metadata.get(
                    "choice_shuffle",
                    {"enabled": False, "seed": None},
                ),
            }
    else:
        for item in load_aligned_examples(args.aligned_path):
            qa_by_id[item.qa.example_id] = {
                "question": item.qa.question,
                "choices": item.qa.choices,
                "answer_index": item.qa.answer_index,
                "choice_shuffle": item.qa.metadata.get(
                    "choice_shuffle",
                    {"enabled": False, "seed": None},
                ),
            }
    return qa_by_id


def _retrieval_from_row(row: dict) -> RetrievalResult:
    return RetrievalResult(
        example_id=str(row["example_id"]),
        reader_id=row.get("reader_id"),
        condition=str(row["condition"]),
        ranked_chunks=[(str(chunk_id), float(score)) for chunk_id, score in row["ranked_chunks"]],
        metadata=dict(row.get("metadata", {})),
    )


def _evidence_chunks(retrieval: RetrievalResult) -> list[Chunk]:
    chunks = []
    for index, evidence in enumerate(retrieval.metadata.get("evidence", [])):
        chunks.append(
            Chunk(
                chunk_id=str(evidence["chunk_id"]),
                paragraph_id=str(retrieval.metadata.get("paragraph_id", "")),
                text=str(evidence["text"]),
                char_start=0,
                char_end=len(str(evidence["text"])),
                token_indices=None,
            )
        )
    if chunks:
        return chunks
    return [
        Chunk(
            chunk_id=chunk_id,
            paragraph_id="",
            text=chunk_id,
            char_start=0,
            char_end=len(chunk_id),
            token_indices=None,
        )
        for chunk_id, _ in retrieval.ranked_chunks
    ]


def _prediction_key(example_id: str, reader_id: str | None, condition: str) -> str:
    return json.dumps([condition, example_id, reader_id], ensure_ascii=False)


def _prediction_key_from_row(row: dict[str, Any]) -> str:
    return _prediction_key(str(row["example_id"]), row.get("reader_id"), str(row["condition"]))


def _prediction_key_from_record(record: PredictionRecord) -> str:
    return _prediction_key(record.example_id, record.reader_id, record.condition)


def _append_jsonl(path: Path, rows: list[PredictionRecord]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            json.dump(to_jsonable(row), fh, ensure_ascii=False)
            fh.write("\n")
        fh.flush()


def _existing_prediction_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    for row in read_jsonl(path):
        keys.add(
            _prediction_key(
                str(row["example_id"]),
                row.get("reader_id"),
                str(row["condition"]),
            )
        )
    return keys


def _validate_resume_choice_config(path: Path, args: argparse.Namespace) -> None:
    if not args.resume or not path.exists():
        return
    first_row: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                first_row = json.loads(line)
                break
    if first_row is None:
        return
    actual = dict(first_row.get("metadata", {}).get("choice_shuffle", {}))
    actual_enabled = bool(actual.get("enabled", False))
    actual_seed = actual.get("seed")
    expected_seed = args.choice_seed if args.shuffle_choices else None
    if actual_enabled != args.shuffle_choices or actual_seed != expected_seed:
        raise ValueError(
            "Existing predictions were created with a different choice-shuffle config. "
            "Use --overwrite or a new --artifacts-dir before rerunning generation."
        )


def _score_prediction(
    generator,
    prompt: str,
    choices: list[str],
    args: argparse.Namespace,
) -> tuple[int | None, str]:
    if args.scoring_mode == "loglik":
        scores = score_answer_options(generator, prompt + " ", choices)
        predicted_index = int(scores.argmax())
        raw_output = json.dumps({"scores": scores.tolist(), "predicted_index": predicted_index})
    else:
        raw_output = generate_answer(generator, prompt, args.max_new_tokens, args.temperature)
        predicted_index = parse_choice(raw_output, len(choices))
    return predicted_index, raw_output


def _progress_iter(items: list[Any], args: argparse.Namespace):
    if args.disable_progress:
        return items
    from tqdm import tqdm

    return tqdm(items, total=len(items), desc="generation", unit="record")


def _write_progress(
    path: Path,
    args: argparse.Namespace,
    device: str | None,
    requested_records: int,
    completed_records: int,
    skipped_existing: int,
    newly_written: int,
    done: bool,
) -> None:
    write_json(
        path,
        {
            "generator_name": args.generator_name,
            "device": device,
            "scoring_mode": args.scoring_mode,
            "shuffle_choices": args.shuffle_choices,
            "choice_seed": args.choice_seed if args.shuffle_choices else None,
            "requested_records": requested_records,
            "completed_records": completed_records,
            "skipped_existing": skipped_existing,
            "newly_written": newly_written,
            "pending_records": max(requested_records - completed_records, 0),
            "done": done,
        },
    )


def main() -> None:
    args = parse_args()
    if args.flush_every <= 0:
        raise ValueError("--flush-every must be positive.")
    qa_by_id = _load_qa_by_id(args)
    retrieval_rows = read_jsonl(args.retrieval_path)
    if args.conditions is not None:
        allowed = set(args.conditions)
        retrieval_rows = [row for row in retrieval_rows if row.get("condition") in allowed]
    if args.max_records is not None:
        retrieval_rows = retrieval_rows[: args.max_records]
    if not retrieval_rows and not args.include_bare_model:
        raise ValueError(f"No retrieval rows found at {args.retrieval_path}.")

    prediction_dir = args.artifacts_dir / "predictions"
    prediction_path = prediction_dir / "predictions.jsonl"
    progress_path = prediction_dir / "prediction_progress.json"
    summary_path = prediction_dir / "prediction_summary.json"
    if args.overwrite or not args.resume:
        prediction_path.unlink(missing_ok=True)
    _validate_resume_choice_config(prediction_path, args)

    completed_keys = _existing_prediction_keys(prediction_path) if args.resume else set()
    skipped_existing = 0
    bare_example_ids = sorted(qa_by_id) if args.include_bare_model else []
    requested_records = len(bare_example_ids) + len(retrieval_rows)

    pending_bare_ids: list[str] = []
    if args.include_bare_model:
        for example_id in bare_example_ids:
            key = _prediction_key(example_id, None, "bare_model")
            if key in completed_keys:
                skipped_existing += 1
            else:
                pending_bare_ids.append(example_id)

    pending_retrieval_rows: list[dict[str, Any]] = []
    for row in retrieval_rows:
        key = _prediction_key_from_row(row)
        if key in completed_keys:
            skipped_existing += 1
        else:
            pending_retrieval_rows.append(row)

    completed_records = skipped_existing
    newly_written = 0
    if not pending_bare_ids and not pending_retrieval_rows:
        _write_progress(
            progress_path,
            args,
            device=None,
            requested_records=requested_records,
            completed_records=min(completed_records, requested_records),
            skipped_existing=skipped_existing,
            newly_written=newly_written,
            done=True,
        )
        summary = {
            "generator_name": args.generator_name,
            "device": None,
            "scoring_mode": args.scoring_mode,
            "shuffle_choices": args.shuffle_choices,
            "choice_seed": args.choice_seed if args.shuffle_choices else None,
            "prediction_records": min(completed_records, requested_records),
            "requested_records": requested_records,
            "skipped_existing": skipped_existing,
            "newly_written": newly_written,
            "predictions_path": str(prediction_path),
            "complete": True,
        }
        write_json(summary_path, summary)
        print(json.dumps(summary, indent=2))
        return

    generator = load_llama_generator(args.generator_name, args.device, args.cache_dir, args.dtype)
    buffer: list[PredictionRecord] = []

    def flush(done: bool = False) -> None:
        nonlocal buffer, newly_written, completed_records
        if buffer:
            _append_jsonl(prediction_path, buffer)
            newly_written += len(buffer)
            completed_records += len(buffer)
            buffer = []
        _write_progress(
            progress_path,
            args,
            device=generator.device,
            requested_records=requested_records,
            completed_records=min(completed_records, requested_records),
            skipped_existing=skipped_existing,
            newly_written=newly_written,
            done=done,
        )

    for example_id in _progress_iter(pending_bare_ids, args):
        qa = qa_by_id[example_id]
        prompt = format_bare_mcqa_prompt(qa["question"], qa["choices"])
        predicted_index, raw_output = _score_prediction(generator, prompt, qa["choices"], args)
        record = PredictionRecord(
            example_id=example_id,
            reader_id=None,
            condition="bare_model",
            prompt=prompt,
            raw_output=raw_output,
            predicted_index=predicted_index,
            gold_index=qa["answer_index"],
            metadata={
                "scoring_mode": args.scoring_mode,
                "choice_shuffle": qa["choice_shuffle"],
            },
        )
        completed_keys.add(_prediction_key_from_record(record))
        buffer.append(record)
        if len(buffer) >= args.flush_every:
            flush(done=False)

    for row in _progress_iter(pending_retrieval_rows, args):
        retrieval = _retrieval_from_row(row)
        if retrieval.example_id not in qa_by_id:
            raise KeyError(f"Missing QA example for retrieval {retrieval.example_id!r}.")
        qa = qa_by_id[retrieval.example_id]
        prompt = format_mcqa_prompt(qa["question"], qa["choices"], _evidence_chunks(retrieval))
        predicted_index, raw_output = _score_prediction(generator, prompt, qa["choices"], args)
        record = PredictionRecord(
            example_id=retrieval.example_id,
            reader_id=retrieval.reader_id,
            condition=retrieval.condition,
            prompt=prompt,
            raw_output=raw_output,
            predicted_index=predicted_index,
            gold_index=qa["answer_index"],
            metadata={
                "scoring_mode": args.scoring_mode,
                "choice_shuffle": qa["choice_shuffle"],
            },
        )
        completed_keys.add(_prediction_key_from_record(record))
        buffer.append(record)
        if len(buffer) >= args.flush_every:
            flush(done=False)

    flush(done=True)
    summary = {
        "generator_name": args.generator_name,
        "device": generator.device,
        "scoring_mode": args.scoring_mode,
        "shuffle_choices": args.shuffle_choices,
        "choice_seed": args.choice_seed if args.shuffle_choices else None,
        "prediction_records": min(completed_records, requested_records),
        "requested_records": requested_records,
        "skipped_existing": skipped_existing,
        "newly_written": newly_written,
        "predictions_path": str(prediction_path),
        "complete": completed_records >= requested_records,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
