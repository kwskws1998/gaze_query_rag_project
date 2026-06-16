import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _write_aligned_fixture(path: Path) -> None:
    words = ["Alpha", "beta.", "Gamma", "delta."]
    reader_gaze = {}
    for reader_id, offset in [("r1", 1.0), ("r2", 2.0)]:
        reader_gaze[reader_id] = [
            {
                "reader_id": reader_id,
                "paragraph_id": "p1",
                "word_index": index + 1,
                "word": word,
                "trt": float(index + offset),
                "fixation_count": 1.0,
                "skip": 0.0,
                "metadata": {},
            }
            for index, word in enumerate(words)
        ]
    row = {
        "qa": {
            "example_id": "q1",
            "paragraph_id": "p1",
            "paragraph_text": "Alpha beta. Gamma delta.",
            "question": "What happened?",
            "choices": ["A", "B", "C", "D"],
            "answer_index": 0,
            "metadata": {},
        },
        "reader_gaze": reader_gaze,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def _records(path: Path) -> list[dict]:
    data = np.load(path, allow_pickle=True)
    return json.loads(str(data["records_json"].item()))


def test_build_embeddings_stores_mean_gaze_once_per_example(tmp_path: Path) -> None:
    aligned_path = tmp_path / "aligned_examples.jsonl"
    artifacts_dir = tmp_path / "artifacts"
    _write_aligned_fixture(aligned_path)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_embeddings.py"),
            "--aligned-path",
            str(aligned_path),
            "--artifacts-dir",
            str(artifacts_dir),
            "--encoder-backend",
            "hash",
            "--hash-dim",
            "16",
            "--chunk-strategy",
            "sentence",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    text_records = _records(artifacts_dir / "embeddings/text_chunks.npz")
    mean_records = _records(artifacts_dir / "embeddings/gaze_chunks_mean.npz")
    actual_records = _records(artifacts_dir / "embeddings/gaze_chunks_actual.npz")

    assert len(mean_records) == len(text_records)
    assert {record["reader_id"] for record in mean_records} == {None}
    assert {record["reader_id"] for record in actual_records} == {"r1", "r2"}
    assert len(actual_records) == 2 * len(text_records)
