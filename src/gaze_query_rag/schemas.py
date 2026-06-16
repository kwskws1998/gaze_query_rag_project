from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class SchemaInferenceError(ValueError):
    pass


@dataclass(frozen=True)
class QAExample:
    example_id: str
    paragraph_id: str
    paragraph_text: str
    question: str
    choices: list[str]
    answer_index: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GazeRecord:
    reader_id: str
    paragraph_id: str
    word_index: int
    word: str
    trt: float
    fixation_count: float | None = None
    skip: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AlignedExample:
    qa: QAExample
    reader_gaze: dict[str, list[GazeRecord]]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    paragraph_id: str
    text: str
    char_start: int
    char_end: int
    token_indices: list[int] | None = None


@dataclass(frozen=True)
class TokenEncoding:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    tokens: list[str]
    offset_mapping: list[tuple[int, int]] | None
    hidden_states: np.ndarray


@dataclass(frozen=True)
class RetrievalResult:
    example_id: str
    reader_id: str | None
    condition: str
    ranked_chunks: list[tuple[str, float]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionRecord:
    example_id: str
    reader_id: str | None
    condition: str
    prompt: str
    raw_output: str
    predicted_index: int | None
    gold_index: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EncoderBundle:
    tokenizer: Any
    model: Any
    device: str
    hidden_size: int


@dataclass(frozen=True)
class GeneratorBundle:
    tokenizer: Any
    model: Any
    device: str
    model_name: str


@dataclass(frozen=True)
class EmbeddingIndex:
    ids: list[str]
    matrix: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
