#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gaze_query_rag.generation.answer_parser import parse_choice
from gaze_query_rag.generation.llama import generate_answer, load_llama_generator, score_answer_options
from gaze_query_rag.generation.prompt import format_bare_mcqa_prompt, format_mcqa_prompt
from gaze_query_rag.data.hf_onestop_qa import load_local_onestop_qa_json, load_onestop_qa
from gaze_query_rag.io.artifacts import load_aligned_examples, read_jsonl, write_json, write_jsonl
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
    return parser.parse_args()


def _load_qa_by_id(args: argparse.Namespace) -> dict[str, dict]:
    qa_by_id = {}
    if args.qa_json_path is not None or args.use_hf:
        examples = (
            load_onestop_qa(args.qa_dataset_name, args.qa_split, cache_dir=None)
            if args.use_hf
            else load_local_onestop_qa_json(args.qa_json_path)
        )
        for item in examples:
            qa_by_id[item.example_id] = {
                "question": item.question,
                "choices": item.choices,
                "answer_index": item.answer_index,
            }
    else:
        for item in load_aligned_examples(args.aligned_path):
            qa_by_id[item.qa.example_id] = {
                "question": item.qa.question,
                "choices": item.qa.choices,
                "answer_index": item.qa.answer_index,
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


def main() -> None:
    args = parse_args()
    qa_by_id = _load_qa_by_id(args)
    retrieval_rows = read_jsonl(args.retrieval_path)
    if args.conditions is not None:
        allowed = set(args.conditions)
        retrieval_rows = [row for row in retrieval_rows if row.get("condition") in allowed]
    if args.max_records is not None:
        retrieval_rows = retrieval_rows[: args.max_records]
    if not retrieval_rows and not args.include_bare_model:
        raise ValueError(f"No retrieval rows found at {args.retrieval_path}.")

    generator = load_llama_generator(args.generator_name, args.device, args.cache_dir, args.dtype)
    predictions: list[PredictionRecord] = []
    if args.include_bare_model:
        for example_id in sorted(qa_by_id):
            qa = qa_by_id[example_id]
            prompt = format_bare_mcqa_prompt(qa["question"], qa["choices"])
            if args.scoring_mode == "loglik":
                scores = score_answer_options(generator, prompt + " ", qa["choices"])
                predicted_index = int(scores.argmax())
                raw_output = json.dumps({"scores": scores.tolist(), "predicted_index": predicted_index})
            else:
                raw_output = generate_answer(
                    generator, prompt, args.max_new_tokens, args.temperature
                )
                predicted_index = parse_choice(raw_output, len(qa["choices"]))
            predictions.append(
                PredictionRecord(
                    example_id=example_id,
                    reader_id=None,
                    condition="bare_model",
                    prompt=prompt,
                    raw_output=raw_output,
                    predicted_index=predicted_index,
                    gold_index=qa["answer_index"],
                    metadata={"scoring_mode": args.scoring_mode},
                )
            )
    for row in retrieval_rows:
        retrieval = _retrieval_from_row(row)
        if retrieval.example_id not in qa_by_id:
            raise KeyError(f"Missing QA example for retrieval {retrieval.example_id!r}.")
        qa = qa_by_id[retrieval.example_id]
        prompt = format_mcqa_prompt(qa["question"], qa["choices"], _evidence_chunks(retrieval))
        if args.scoring_mode == "loglik":
            scores = score_answer_options(generator, prompt + " ", qa["choices"])
            predicted_index = int(scores.argmax())
            raw_output = json.dumps({"scores": scores.tolist(), "predicted_index": predicted_index})
        else:
            raw_output = generate_answer(
                generator, prompt, args.max_new_tokens, args.temperature
            )
            predicted_index = parse_choice(raw_output, len(qa["choices"]))
        predictions.append(
            PredictionRecord(
                example_id=retrieval.example_id,
                reader_id=retrieval.reader_id,
                condition=retrieval.condition,
                prompt=prompt,
                raw_output=raw_output,
                predicted_index=predicted_index,
                gold_index=qa["answer_index"],
                metadata={"scoring_mode": args.scoring_mode},
            )
        )

    prediction_dir = args.artifacts_dir / "predictions"
    prediction_path = prediction_dir / "predictions.jsonl"
    write_jsonl(prediction_path, predictions)
    summary = {
        "generator_name": args.generator_name,
        "device": generator.device,
        "scoring_mode": args.scoring_mode,
        "prediction_records": len(predictions),
        "predictions_path": str(prediction_path),
    }
    write_json(prediction_dir / "prediction_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
