from __future__ import annotations

from pathlib import Path

from gaze_query_rag.data.hf_onestop_qa import load_local_onestop_qa_json, shuffle_qa_choices
from gaze_query_rag.schemas import QAExample


def _example() -> QAExample:
    return QAExample(
        example_id="q1",
        paragraph_id="p1",
        paragraph_text="Paragraph.",
        question="Question?",
        choices=["correct", "wrong-a", "wrong-b", "wrong-c"],
        answer_index=0,
    )


def test_choice_shuffle_is_deterministic() -> None:
    first = shuffle_qa_choices(_example(), seed=13)
    second = shuffle_qa_choices(_example(), seed=13)

    assert first.choices == second.choices
    assert first.answer_index == second.answer_index


def test_choice_shuffle_remaps_gold_index() -> None:
    shuffled = shuffle_qa_choices(_example(), seed=13)

    assert shuffled.choices[shuffled.answer_index] == "correct"
    assert sorted(shuffled.choices) == ["correct", "wrong-a", "wrong-b", "wrong-c"]
    assert shuffled.metadata["choice_shuffle"]["original_answer_index"] == 0


def test_local_onestop_choice_shuffle_breaks_all_a_bias() -> None:
    path = Path("resources/onestop_qa.json")
    if not path.exists():
        return

    examples = load_local_onestop_qa_json(path, shuffle_choices=True, choice_seed=13)
    answer_indices = {example.answer_index for example in examples}

    assert 0 in answer_indices
    assert len(answer_indices) > 1
