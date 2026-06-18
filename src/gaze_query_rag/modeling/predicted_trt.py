from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from gaze_query_rag.modeling.gaze_features import compute_gaze_distribution, word_trt_to_token_trt
from gaze_query_rag.modeling.gaze_query_attention import build_gaze_view_chunk_embeddings
from gaze_query_rag.schemas import Chunk


PREDICTED_TRT_GAZE_CONDITION = "predicted_trt_gaze"
SKBOY_ET_REPO_ID = "skboy/et_prediction_2"
SKBOY_ET_WEIGHTS = "et_predictor2_seed123.safetensors"
FEATURE_NAMES = ["nFix", "FFD", "GPT", "TRT", "fixProp"]


@dataclass(frozen=True)
class PredictedTRTWord:
    word: str
    trt: float
    features: dict[str, float]


class TRTPredictor(Protocol):
    def predict_words(self, text: str) -> list[PredictedTRTWord]:
        ...


class SkboyTRTPredictor:
    def __init__(
        self,
        repo_id: str = SKBOY_ET_REPO_ID,
        weights_filename: str = SKBOY_ET_WEIGHTS,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        from huggingface_hub import hf_hub_download

        download_kwargs = {
            "repo_id": repo_id,
            "local_files_only": local_files_only,
        }
        if cache_dir is not None:
            download_kwargs["cache_dir"] = str(cache_dir)
        model_py = hf_hub_download(filename="model.py", **download_kwargs)
        weights = hf_hub_download(filename=weights_filename, **download_kwargs)
        module = _load_module(Path(model_py))
        self.predictor = module.FixationsPredictor2(weights)

    def predict_words(self, text: str) -> list[PredictedTRTWord]:
        features, words = self.predictor.predict_raw_text(text)
        feature_array = np.asarray(features, dtype=np.float64)
        if feature_array.ndim != 2 or feature_array.shape[1] < len(FEATURE_NAMES):
            raise ValueError("ET predictor returned an unexpected feature matrix shape.")
        if len(words) != feature_array.shape[0]:
            raise ValueError("ET predictor returned different word and feature counts.")
        rows: list[PredictedTRTWord] = []
        trt_index = FEATURE_NAMES.index("TRT")
        for word, feature_row in zip(words, feature_array):
            feature_dict = {
                name: float(feature_row[index]) for index, name in enumerate(FEATURE_NAMES)
            }
            rows.append(
                PredictedTRTWord(
                    word=str(word),
                    trt=float(feature_row[trt_index]),
                    features=feature_dict,
                )
            )
        if not rows:
            raise ValueError("ET predictor returned no words.")
        return rows


class HeuristicTRTPredictor:
    def predict_words(self, text: str) -> list[PredictedTRTWord]:
        rows: list[PredictedTRTWord] = []
        for raw_word in re.findall(r"\S+", text):
            cleaned = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", raw_word)
            score = max(0.0, len(cleaned) / 10.0)
            rows.append(
                PredictedTRTWord(
                    word=raw_word,
                    trt=score,
                    features={
                        "nFix": 1.0 + score,
                        "FFD": score,
                        "GPT": score,
                        "TRT": score,
                        "fixProp": min(1.0, score),
                    },
                )
            )
        if not rows:
            raise ValueError("Cannot predict TRT for empty text.")
        return rows


def load_trt_predictor(
    backend: str,
    repo_id: str = SKBOY_ET_REPO_ID,
    weights_filename: str = SKBOY_ET_WEIGHTS,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> TRTPredictor:
    if backend == "skboy":
        return SkboyTRTPredictor(
            repo_id=repo_id,
            weights_filename=weights_filename,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    if backend == "heuristic":
        return HeuristicTRTPredictor()
    raise ValueError(f"Unsupported predicted TRT backend: {backend}")


def predicted_words_to_token_trt(
    predicted_words: list[PredictedTRTWord],
    word_to_token_indices: list[list[int]],
    num_tokens: int,
) -> np.ndarray:
    word_trt = np.asarray([row.trt for row in predicted_words], dtype=np.float64)
    return word_trt_to_token_trt(word_trt, word_to_token_indices, num_tokens)


def build_predicted_trt_chunk_embeddings(
    hidden: np.ndarray,
    token_trt: np.ndarray,
    chunks: list[Chunk],
) -> dict[str, np.ndarray]:
    gaze = compute_gaze_distribution(token_trt)
    return build_gaze_view_chunk_embeddings(hidden, gaze, chunks)


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("skboy_et_prediction_2_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import ET predictor module from {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
