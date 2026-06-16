import numpy as np

from gaze_query_rag.modeling.gaze_query_attention import (
    build_gaze_view_chunk_embeddings,
    build_text_chunk_embeddings,
    gaze_query_attention,
    gaze_weight_tokens,
    pool_chunk_embedding,
)
from gaze_query_rag.schemas import Chunk


def test_gaze_query_attention_output_shape_is_correct() -> None:
    hidden = np.arange(12, dtype=np.float64).reshape(4, 3)
    gaze = np.full(4, 0.25)

    attended = gaze_query_attention(hidden, gaze)

    assert attended.shape == hidden.shape


def test_gaze_weight_tokens_scales_rows() -> None:
    hidden = np.ones((3, 2))
    gaze = np.array([0.1, 0.2, 0.7])

    weighted = gaze_weight_tokens(hidden, gaze)

    assert np.allclose(weighted, np.array([[0.1, 0.1], [0.2, 0.2], [0.7, 0.7]]))


def test_chunk_pooling_normalizes_embeddings() -> None:
    hidden = np.array([[3.0, 0.0], [0.0, 4.0]])

    pooled = pool_chunk_embedding(hidden, [0, 1], normalize=True)

    assert np.isclose(np.linalg.norm(pooled), 1.0)


def test_build_chunk_embeddings_uses_token_indices() -> None:
    hidden = np.eye(4)
    gaze = np.full(4, 0.25)
    chunks = [
        Chunk("c1", "p1", "a", 0, 1, token_indices=[0, 1]),
        Chunk("c2", "p1", "b", 2, 3, token_indices=[2, 3]),
    ]

    text_embeddings = build_text_chunk_embeddings(hidden, chunks)
    gaze_embeddings = build_gaze_view_chunk_embeddings(hidden, gaze, chunks)

    assert set(text_embeddings) == {"c1", "c2"}
    assert set(gaze_embeddings) == {"c1", "c2"}
    assert np.isclose(np.linalg.norm(text_embeddings["c1"]), 1.0)
