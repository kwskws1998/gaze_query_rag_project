import numpy as np

from gaze_query_rag.modeling.skip_attention import (
    build_skip_hard_chunk_embeddings,
    word_skip_to_token_query_weights,
)
from gaze_query_rag.schemas import Chunk


def test_word_skip_to_token_query_weights_uses_hard_inspect_mask() -> None:
    word_skip = np.array([0.0, 1.0, 0.0])
    word_to_token = [[0, 1], [2], [3]]

    weights = word_skip_to_token_query_weights(word_skip, word_to_token, num_tokens=5)

    np.testing.assert_allclose(weights, np.array([1.0, 1.0, 0.0, 1.0, 0.0]))


def test_build_skip_hard_chunk_embeddings_returns_normalized_vectors() -> None:
    hidden = np.eye(3, dtype=np.float64)
    weights = np.array([1.0, 0.0, 1.0])
    chunks = [
        Chunk("c1", "p1", "a b", 0, 3, token_indices=[0, 1]),
        Chunk("c2", "p1", "c", 4, 5, token_indices=[2]),
    ]

    embeddings = build_skip_hard_chunk_embeddings(hidden, weights, chunks)

    assert set(embeddings) == {"c1", "c2"}
    for vector in embeddings.values():
        assert np.isclose(np.linalg.norm(vector), 1.0)
