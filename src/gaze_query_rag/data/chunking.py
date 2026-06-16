from __future__ import annotations

import re

from gaze_query_rag.schemas import Chunk


def _word_spans(text: str) -> list[re.Match[str]]:
    return list(re.finditer(r"\S+", text))


def _chunk_id(paragraph_id: str, index: int) -> str:
    return f"{paragraph_id}:chunk={index}"


def _make_chunk(paragraph_id: str, text: str, start: int, end: int, index: int) -> Chunk:
    return Chunk(
        chunk_id=_chunk_id(paragraph_id, index),
        paragraph_id=paragraph_id,
        text=text[start:end].strip(),
        char_start=start,
        char_end=end,
        token_indices=None,
    )


def _fixed_word_chunks(
    paragraph_id: str, text: str, max_words: int, stride: int, start_index: int = 0
) -> list[Chunk]:
    if max_words <= 0:
        raise ValueError("max_words must be positive.")
    if stride < 0 or stride >= max_words:
        raise ValueError("stride must satisfy 0 <= stride < max_words.")
    words = _word_spans(text)
    if not words:
        return []
    step = max_words - stride if stride else max_words
    chunks: list[Chunk] = []
    chunk_index = start_index
    for start_word in range(0, len(words), step):
        end_word = min(start_word + max_words, len(words))
        start = words[start_word].start()
        end = words[end_word - 1].end()
        chunks.append(_make_chunk(paragraph_id, text, start, end, chunk_index))
        chunk_index += 1
        if end_word == len(words):
            break
    return chunks


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"[.!?]+[\"')\]]*(?=\s+|$)", text):
        end = match.end()
        if text[start:end].strip():
            spans.append((start, end))
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text)))
    return spans


def chunk_paragraph(
    paragraph_id: str,
    text: str,
    strategy: str,
    max_words: int,
    stride: int = 0,
) -> list[Chunk]:
    if strategy == "whole_paragraph":
        stripped_start = len(text) - len(text.lstrip())
        stripped_end = len(text.rstrip())
        if stripped_start >= stripped_end:
            return []
        return [_make_chunk(paragraph_id, text, stripped_start, stripped_end, 0)]
    if strategy == "fixed_words":
        return _fixed_word_chunks(paragraph_id, text, max_words=max_words, stride=stride)
    if strategy != "sentence":
        raise ValueError(f"Unsupported chunking strategy: {strategy}")

    chunks: list[Chunk] = []
    for start, end in _sentence_spans(text):
        sentence = text[start:end]
        if len(_word_spans(sentence)) <= max_words:
            chunks.append(_make_chunk(paragraph_id, text, start, end, len(chunks)))
            continue
        local_chunks = _fixed_word_chunks(
            paragraph_id, sentence, max_words=max_words, stride=stride, start_index=len(chunks)
        )
        for local in local_chunks:
            chunks.append(
                Chunk(
                    chunk_id=_chunk_id(paragraph_id, len(chunks)),
                    paragraph_id=paragraph_id,
                    text=local.text,
                    char_start=start + local.char_start,
                    char_end=start + local.char_end,
                    token_indices=None,
                )
            )
    return chunks
