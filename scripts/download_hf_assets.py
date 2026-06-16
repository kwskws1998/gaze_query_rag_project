#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_ENCODER = "intfloat/e5-large-v2"
DEFAULT_SMOKE_GENERATOR = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_LLAMA3 = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_DATASET = "malmaud/onestop_qa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-name", default=DEFAULT_ENCODER)
    parser.add_argument("--generator-name", default=DEFAULT_SMOKE_GENERATOR)
    parser.add_argument("--include-generator", action="store_true", default=True)
    parser.add_argument("--skip-generator", action="store_true")
    parser.add_argument("--include-llama3", action="store_true")
    parser.add_argument("--llama3-name", default=DEFAULT_LLAMA3)
    parser.add_argument("--extra-model", action="append", default=[])
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--include-dataset", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--revision", default=None)
    return parser.parse_args()


def _snapshot_model(repo_id: str, cache_dir: Path | None, revision: str | None) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    local_dir = snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_dir) if cache_dir else None,
        revision=revision,
        token=token,
    )
    return {"repo_id": repo_id, "local_dir": local_dir}


def _download_dataset(dataset_name: str, cache_dir: Path | None) -> dict[str, str]:
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    dataset = load_dataset(
        dataset_name,
        cache_dir=str(cache_dir) if cache_dir else None,
        token=token,
    )
    return {"dataset_name": dataset_name, "splits": ",".join(dataset.keys())}


def main() -> None:
    args = parse_args()
    downloaded: dict[str, list[dict[str, str]]] = {"models": [], "datasets": []}

    model_names = [args.encoder_name]
    if args.include_generator and not args.skip_generator:
        model_names.append(args.generator_name)
    if args.include_llama3:
        model_names.append(args.llama3_name)
    model_names.extend(args.extra_model)

    seen: set[str] = set()
    for model_name in model_names:
        if model_name in seen:
            continue
        seen.add(model_name)
        downloaded["models"].append(_snapshot_model(model_name, args.cache_dir, args.revision))

    if args.include_dataset:
        downloaded["datasets"].append(_download_dataset(args.dataset_name, args.cache_dir))

    print(json.dumps(downloaded, indent=2))


if __name__ == "__main__":
    main()
