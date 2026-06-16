from __future__ import annotations

from gaze_query_rag.schemas import Chunk


def format_choice_block(choices: list[str]) -> str:
    if not choices:
        raise ValueError("choices must not be empty.")
    if len(choices) > 26:
        raise ValueError("At most 26 choices are supported.")
    lines = []
    for index, choice in enumerate(choices):
        label = chr(ord("A") + index)
        lines.append(f"{label}. {choice}")
    return "\n".join(lines)


def format_mcqa_prompt(question: str, choices: list[str], evidence_chunks: list[Chunk]) -> str:
    if not question.strip():
        raise ValueError("question must not be empty.")
    if not evidence_chunks:
        evidence_text = "No retrieved evidence."
    else:
        evidence_text = "\n\n".join(
            f"[Evidence {index + 1}]\n{chunk.text.strip()}"
            for index, chunk in enumerate(evidence_chunks)
        )
    return (
        "Answer the multiple-choice question using only the retrieved evidence.\n"
        "Return exactly one choice label.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Choices:\n{format_choice_block(choices)}\n\n"
        f"Retrieved evidence:\n{evidence_text}\n\n"
        "Answer:"
    )


def format_bare_mcqa_prompt(question: str, choices: list[str]) -> str:
    if not question.strip():
        raise ValueError("question must not be empty.")
    return (
        "Answer the multiple-choice question.\n"
        "Return exactly one choice label.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Choices:\n{format_choice_block(choices)}\n\n"
        "Answer:"
    )
