import numpy as np

from gaze_query_rag.evaluation.rag_rewardbench_compression import (
    QUERY_X_PREDICTED_ET_COMPRESSION,
    combine_chunk_scores,
    compress_context_by_scores,
    split_context_chunks,
)


def test_split_context_chunks_splits_long_sentence_by_word_budget() -> None:
    chunks = split_context_chunks("one two three four five.", max_words=2)

    assert [chunk.text for chunk in chunks] == ["one two", "three four", "five."]
    assert [chunk.index for chunk in chunks] == [0, 1, 2]


def test_compress_context_by_scores_preserves_original_order() -> None:
    text = "Alpha first. Beta second. Gamma third."
    chunks = split_context_chunks(text, max_words=10)
    scores = np.array([0.1, 0.9, 0.8])

    compressed = compress_context_by_scores(
        text,
        chunks,
        scores,
        budget_tokens=4,
        token_count_fn=lambda value: len(value.split()),
    )

    assert [chunk.text for chunk in compressed.selected_chunks] == ["Beta second.", "Gamma third."]
    assert compressed.text == "Beta second.\n\nGamma third."


def test_query_x_predicted_et_compression_keeps_query_as_anchor() -> None:
    scores = combine_chunk_scores(
        QUERY_X_PREDICTED_ET_COMPRESSION,
        query_scores=np.array([1.0, 0.1]),
        trt_scores=np.array([0.1, 1.0]),
        alpha=1.0,
        beta=0.1,
        tau_q=0.1,
        tau_g=0.1,
    )

    assert np.isclose(scores.sum(), 1.0)
    assert scores[0] > scores[1]
