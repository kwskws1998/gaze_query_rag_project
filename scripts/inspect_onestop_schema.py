#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gaze_query_rag.data.hf_onestop_qa import load_local_onestop_qa_json
from gaze_query_rag.data.onestop_gaze import inspect_onestop_ia_schema
from gaze_query_rag.io.artifacts import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ia-path",
        type=Path,
        default=ROOT
        / "OneStop-Eye-Movements/data/OneStop/osfstorage-archive (1)/ia_Paragraph.csv.zip",
    )
    parser.add_argument(
        "--qa-json-path",
        type=Path,
        default=ROOT / "OneStop-Eye-Movements/data_preprocessing/onestop_qa.json",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts")
    return parser.parse_args()


def inspect_local_qa(path: Path) -> dict:
    examples = load_local_onestop_qa_json(path)
    first = examples[0] if examples else None
    return {
        "path": str(path),
        "example_count": len(examples),
        "columns": [
            "example_id",
            "paragraph_id",
            "paragraph_text",
            "question",
            "choices",
            "answer_index",
            "metadata",
        ],
        "first_example": None
        if first is None
        else {
            "example_id": first.example_id,
            "paragraph_id": first.paragraph_id,
            "paragraph_text_prefix": first.paragraph_text[:240],
            "question": first.question,
            "choices": first.choices,
            "answer_index": first.answer_index,
            "metadata_keys": sorted(first.metadata.keys()),
        },
    }


def main() -> None:
    args = parse_args()
    schema_dir = args.artifacts_dir / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)

    ia_schema = inspect_onestop_ia_schema(args.ia_path)
    write_json(schema_dir / "onestop_ia_columns.json", ia_schema)
    pd.DataFrame(ia_schema["sample_rows"]).to_csv(schema_dir / "onestop_ia_sample.csv", index=False)

    qa_schema = inspect_local_qa(args.qa_json_path)
    write_json(schema_dir / "onestop_qa_columns.json", qa_schema)

    print(
        json.dumps(
            {
                "ia_columns": ia_schema["column_count"],
                "ia_rows": ia_schema["row_count"],
                "qa_examples": qa_schema["example_count"],
                "schema_dir": str(schema_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
