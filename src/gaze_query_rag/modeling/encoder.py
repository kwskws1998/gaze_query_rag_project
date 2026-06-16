from __future__ import annotations

import re
import string
from pathlib import Path

import numpy as np

from gaze_query_rag.modeling.devices import resolve_torch_device
from gaze_query_rag.schemas import EncoderBundle, TokenEncoding


PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or not np.isfinite(norm):
        return np.zeros_like(vector, dtype=np.float64)
    return vector / norm


def _mean_pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1)
    return summed / counts


def _adjust_offsets(
    offsets: list[tuple[int, int]], prefix_len: int, original_text_len: int
) -> list[tuple[int, int]]:
    adjusted: list[tuple[int, int]] = []
    for start, end in offsets:
        if start == end or end <= prefix_len:
            adjusted.append((-1, -1))
            continue
        shifted_start = max(0, start - prefix_len)
        shifted_end = min(original_text_len, end - prefix_len)
        if shifted_start >= shifted_end:
            adjusted.append((-1, -1))
        else:
            adjusted.append((shifted_start, shifted_end))
    return adjusted


def load_e5_encoder(
    model_name: str = "intfloat/e5-large-v2",
    device: str = "auto",
    cache_dir: str | Path | None = None,
) -> EncoderBundle:
    import torch
    from transformers import AutoModel, AutoTokenizer

    resolved_device = resolve_torch_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=str(cache_dir) if cache_dir else None, use_fast=True
    )
    model = AutoModel.from_pretrained(model_name, cache_dir=str(cache_dir) if cache_dir else None)
    model.to(resolved_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    hidden_size = int(getattr(model.config, "hidden_size"))
    torch.set_grad_enabled(False)
    return EncoderBundle(
        tokenizer=tokenizer,
        model=model,
        device=str(resolved_device),
        hidden_size=hidden_size,
    )


def encode_passage_tokens(bundle: EncoderBundle, text: str, max_length: int = 512) -> TokenEncoding:
    import torch

    prefixed = PASSAGE_PREFIX + text
    try:
        encoded = bundle.tokenizer(
            prefixed,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        raw_offsets = encoded.pop("offset_mapping")[0].tolist()
        offsets = _adjust_offsets(
            [(int(start), int(end)) for start, end in raw_offsets],
            len(PASSAGE_PREFIX),
            len(text),
        )
    except (NotImplementedError, TypeError):
        encoded = bundle.tokenizer(
            prefixed,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )
        offsets = None

    encoded = {key: value.to(bundle.device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = bundle.model(**encoded)
    hidden = output.last_hidden_state[0].detach().cpu().numpy()
    input_ids = encoded["input_ids"][0].detach().cpu().numpy()
    attention_mask = encoded["attention_mask"][0].detach().cpu().numpy()
    tokens = bundle.tokenizer.convert_ids_to_tokens(input_ids.tolist())
    return TokenEncoding(
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokens=tokens,
        offset_mapping=offsets,
        hidden_states=hidden,
    )


def encode_query(bundle: EncoderBundle, question: str, max_length: int = 128) -> np.ndarray:
    import torch

    encoded = bundle.tokenizer(
        QUERY_PREFIX + question,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors="pt",
    )
    encoded = {key: value.to(bundle.device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = bundle.model(**encoded)
    pooled = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
    return _normalize_vector(pooled[0].detach().cpu().numpy())


def _canonical_word(text: str) -> str:
    replacements = str.maketrans(
        {
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            "–": "-",
            "—": "-",
        }
    )
    normalized = text.translate(replacements).strip()
    return normalized.strip(string.punctuation).lower()


def _word_char_spans(text: str, words: list[str]) -> list[tuple[int, int]]:
    text_spans = list(re.finditer(r"\S+", text))
    cursor = 0
    spans: list[tuple[int, int]] = []
    for word_index, word in enumerate(words):
        target = str(word).strip()
        if not target:
            raise ValueError(f"Empty word at position {word_index}.")
        target_key = _canonical_word(target)
        matched_span: tuple[int, int] | None = None
        for match in text_spans:
            if match.end() < cursor:
                continue
            candidate = match.group(0)
            candidate_key = _canonical_word(candidate)
            if candidate == target:
                matched_span = match.span()
                break
            if target in candidate:
                inner_start = match.start() + candidate.index(target)
                if inner_start >= cursor:
                    matched_span = (inner_start, inner_start + len(target))
                    break
            stripped_target = target.strip(string.punctuation)
            if stripped_target and stripped_target in candidate:
                inner_start = match.start() + candidate.index(stripped_target)
                if inner_start >= cursor:
                    matched_span = (inner_start, inner_start + len(stripped_target))
                    break
            if target in candidate and candidate_key == target_key:
                inner_start = match.start() + candidate.index(target)
                matched_span = (inner_start, inner_start + len(target))
                break
            if candidate_key == target_key:
                matched_span = match.span()
                break
        if matched_span is None:
            preview = text[max(0, cursor - 40) : cursor + 120]
            raise ValueError(
                f"Could not align word {word_index}={target!r} after char {cursor}; "
                f"text preview: {preview!r}"
            )
        spans.append(matched_span)
        cursor = matched_span[1]
    return spans


def build_word_to_token_alignment(
    text: str, words: list[str], token_offsets: list[tuple[int, int]]
) -> list[list[int]]:
    word_spans = _word_char_spans(text, words)
    alignment: list[list[int]] = []
    for word_index, (word_start, word_end) in enumerate(word_spans):
        token_indices: list[int] = []
        for token_index, (token_start, token_end) in enumerate(token_offsets):
            if token_start < 0 or token_end <= token_start:
                continue
            if token_end > word_start and token_start < word_end:
                token_indices.append(token_index)
        if not token_indices:
            raise ValueError(
                f"Word {word_index}={words[word_index]!r} matched chars "
                f"{word_start}:{word_end} but no tokenizer offsets overlapped."
            )
        alignment.append(token_indices)
    return alignment
