import numpy as np

from gaze_query_rag.modeling.mean_gaze import (
    MEAN_GAZE_CONDITION,
    build_mean_gaze_chunk_embeddings,
    build_mean_gaze_vector,
)
from gaze_query_rag.schemas import Chunk


def test_mean_gaze_condition_name_is_stable() -> None:
    assert MEAN_GAZE_CONDITION == "mean_gaze"


def test_build_mean_gaze_vector_averages_reader_distributions() -> None:
    mean = build_mean_gaze_vector(
        {
            "r1": np.array([0.6, 0.4, 0.0]),
            "r2": np.array([0.0, 0.2, 0.8]),
        }
    )

    assert np.allclose(mean, np.array([0.3, 0.3, 0.4]))
    assert np.isclose(mean.sum(), 1.0)


def test_build_mean_gaze_chunk_embeddings_is_reader_order_invariant() -> None:
    hidden = np.eye(3, dtype=np.float64)
    chunks = [
        Chunk("c1", "p1", "a", 0, 1, token_indices=[0, 1]),
        Chunk("c2", "p1", "b", 2, 3, token_indices=[2]),
    ]
    gaze_a = {
        "r1": np.array([0.8, 0.1, 0.1]),
        "r2": np.array([0.2, 0.4, 0.4]),
    }
    gaze_b = {
        "r2": np.array([0.2, 0.4, 0.4]),
        "r1": np.array([0.8, 0.1, 0.1]),
    }

    embeddings_a = build_mean_gaze_chunk_embeddings(hidden, gaze_a, chunks)
    embeddings_b = build_mean_gaze_chunk_embeddings(hidden, gaze_b, chunks)

    assert set(embeddings_a) == {"c1", "c2"}
    assert np.allclose(embeddings_a["c1"], embeddings_b["c1"])
    assert np.allclose(embeddings_a["c2"], embeddings_b["c2"])
    assert np.isclose(np.linalg.norm(embeddings_a["c1"]), 1.0)
