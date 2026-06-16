from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from gaze_query_rag.modeling.gaze_features import compute_mean_gaze
from gaze_query_rag.modeling.gaze_query_attention import build_gaze_view_chunk_embeddings
from gaze_query_rag.schemas import Chunk


MEAN_GAZE_CONDITION = "mean_gaze"


def build_mean_gaze_vector(gaze_by_reader: Mapping[str, np.ndarray]) -> np.ndarray:
    return compute_mean_gaze(dict(gaze_by_reader))


def build_mean_gaze_chunk_embeddings(
    hidden: np.ndarray,
    gaze_by_reader: Mapping[str, np.ndarray],
    chunks: Sequence[Chunk],
) -> dict[str, np.ndarray]:
    mean_gaze = build_mean_gaze_vector(gaze_by_reader)
    return build_gaze_view_chunk_embeddings(hidden, mean_gaze, list(chunks))
