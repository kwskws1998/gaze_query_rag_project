#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-examples", type=int, default=8)
    parser.add_argument("--max-readers-per-example", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--encoder-backend", choices=["e5", "hash"], default="hash")
    parser.add_argument("--run-generation", action="store_true")
    parser.add_argument("--generator-name", default="meta-llama/Meta-Llama-3-8B-Instruct")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    python = sys.executable
    _run(
        [
            python,
            "scripts/inspect_onestop_schema.py",
        ]
    )
    _run(
        [
            python,
            "scripts/build_aligned_dataset.py",
            "--max-examples",
            str(args.max_examples),
            "--max-readers-per-example",
            str(args.max_readers_per_example),
        ]
    )
    _run(
        [
            python,
            "scripts/build_embeddings.py",
            "--encoder-backend",
            args.encoder_backend,
            "--max-examples",
            str(args.max_examples),
        ]
    )
    _run([python, "scripts/run_retrieval.py", "--top-k", str(args.top_k)])
    generation_status = "skipped"
    if args.run_generation:
        _run(
            [
                python,
                "scripts/run_generation.py",
                "--generator-name",
                args.generator_name,
            ]
        )
        generation_status = "completed"
    _run([python, "scripts/run_eval.py"])
    summary = {
        "max_examples": args.max_examples,
        "max_readers_per_example": args.max_readers_per_example,
        "top_k": args.top_k,
        "encoder_backend": args.encoder_backend,
        "generation_status": generation_status,
        "results_summary": str(ROOT / "artifacts/results/summary.json"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
