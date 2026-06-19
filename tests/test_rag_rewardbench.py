import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_rag_rewardbench import _format_prompt_prefix, _row_id, _summarize_predictions


def test_rag_rewardbench_row_id_uses_dataset_id_when_present() -> None:
    assert _row_id(3, {"id": "abc"}) == "abc"
    assert _row_id(3, {"id": ""}) == "3"
    assert _row_id(3, {}) == "3"


def test_rag_rewardbench_prompt_uses_references_and_question() -> None:
    prompt = _format_prompt_prefix(
        {
            "prompt": "## References\nReference [1] Text: Alpha.",
            "question": "What is alpha?",
        }
    )

    assert "Reference [1]" in prompt
    assert "What is alpha?" in prompt
    assert prompt.endswith("Answer:\n")


def test_rag_rewardbench_summary_groups_by_subset() -> None:
    summary, by_subset = _summarize_predictions(
        [
            {"subset": "a", "correct": True, "tie": False, "score_delta": 1.0},
            {"subset": "a", "correct": False, "tie": False, "score_delta": -1.0},
            {"subset": "b", "correct": True, "tie": True, "score_delta": 0.0},
        ]
    )

    assert summary["prediction_records"] == 3
    assert summary["accuracy"] == 2 / 3
    by_subset_map = {row["subset"]: row for row in by_subset}
    assert by_subset_map["a"]["accuracy"] == 0.5
    assert by_subset_map["b"]["accuracy"] == 1.0
