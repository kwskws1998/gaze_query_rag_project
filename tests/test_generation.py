from gaze_query_rag.generation.answer_parser import parse_choice
from gaze_query_rag.generation.prompt import (
    format_bare_mcqa_prompt,
    format_choice_block,
    format_mcqa_prompt,
)
from gaze_query_rag.schemas import Chunk


def test_format_choice_block_labels_choices() -> None:
    assert format_choice_block(["alpha", "beta"]) == "A. alpha\nB. beta"


def test_format_mcqa_prompt_keeps_evidence_separate() -> None:
    prompt = format_mcqa_prompt(
        "What is true?",
        ["One", "Two"],
        [Chunk("c1", "p1", "Evidence text", 0, 13)],
    )

    assert "Question:\nWhat is true?" in prompt
    assert "A. One\nB. Two" in prompt
    assert "[Evidence 1]\nEvidence text" in prompt
    assert prompt.endswith("Answer:")


def test_format_bare_mcqa_prompt_has_no_evidence_block() -> None:
    prompt = format_bare_mcqa_prompt("What is true?", ["One", "Two"])

    assert "Retrieved evidence" not in prompt
    assert "A. One\nB. Two" in prompt


def test_parse_choice_accepts_common_formats() -> None:
    assert parse_choice("A", 4) == 0
    assert parse_choice("(B) because", 4) == 1
    assert parse_choice("The answer is C.", 4) == 2
    assert parse_choice("option D", 4) == 3


def test_parse_choice_returns_none_for_ambiguous_output() -> None:
    assert parse_choice("A or B", 4) is None
    assert parse_choice("No clear answer", 4) is None
