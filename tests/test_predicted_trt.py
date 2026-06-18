import numpy as np

from gaze_query_rag.modeling.predicted_trt import (
    HeuristicTRTPredictor,
    build_predicted_trt_chunk_embeddings,
    load_trt_predictor,
    predicted_words_to_token_trt,
)
from gaze_query_rag.schemas import Chunk


def test_heuristic_trt_predictor_returns_trt_feature() -> None:
    predictor = load_trt_predictor("heuristic")

    rows = predictor.predict_words("Alpha beta.")

    assert isinstance(predictor, HeuristicTRTPredictor)
    assert [row.word for row in rows] == ["Alpha", "beta."]
    assert all(row.trt >= 0.0 for row in rows)
    assert all("TRT" in row.features for row in rows)


def test_predicted_words_to_token_trt_preserves_mass() -> None:
    predictor = HeuristicTRTPredictor()
    predicted_words = predictor.predict_words("Alpha beta")
    alignment = [[0, 1], [2]]

    token_trt = predicted_words_to_token_trt(predicted_words, alignment, num_tokens=3)

    assert np.isclose(token_trt.sum(), sum(row.trt for row in predicted_words))


def test_build_predicted_trt_chunk_embeddings_returns_normalized_vectors() -> None:
    hidden = np.eye(3, dtype=np.float64)
    chunks = [
        Chunk("c1", "p1", "Alpha", 0, 5, token_indices=[0]),
        Chunk("c2", "p1", "beta", 6, 10, token_indices=[1, 2]),
    ]
    token_trt = np.array([1.0, 0.5, 0.25])

    embeddings = build_predicted_trt_chunk_embeddings(hidden, token_trt, chunks)

    assert set(embeddings) == {"c1", "c2"}
    assert np.isclose(np.linalg.norm(embeddings["c1"]), 1.0)
    assert np.isclose(np.linalg.norm(embeddings["c2"]), 1.0)
