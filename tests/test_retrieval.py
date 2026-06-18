import numpy as np

from gaze_query_rag.retrieval.index import build_in_memory_index, search_index
from gaze_query_rag.retrieval.retriever import retrieve_gaze_view, retrieve_text_only
from gaze_query_rag.schemas import Chunk, QAExample
from scripts.run_retrieval import _retrieve_hybrid_group, _retrieve_text_candidate_gaze_rerank_group


def _example() -> QAExample:
    return QAExample(
        example_id="q1",
        paragraph_id="p1",
        paragraph_text="alpha beta",
        question="question",
        choices=["A", "B"],
        answer_index=0,
    )


def _chunks() -> list[Chunk]:
    return [
        Chunk("c1", "p1", "alpha", 0, 5, token_indices=[0]),
        Chunk("c2", "p1", "beta", 6, 10, token_indices=[1]),
        Chunk("c3", "p1", "gamma", 11, 16, token_indices=[2]),
    ]


def test_text_only_retrieval_returns_sorted_scores() -> None:
    embeddings = {
        "c1": np.array([1.0, 0.0]),
        "c2": np.array([0.5, 0.5]),
        "c3": np.array([-1.0, 0.0]),
    }
    index = build_in_memory_index(embeddings)

    results = search_index(index, np.array([1.0, 0.0]), top_k=3)

    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    assert [chunk_id for chunk_id, _ in results] == ["c1", "c2", "c3"]


def test_retrieve_text_only_builds_result_record() -> None:
    embeddings = {
        "c1": np.array([1.0, 0.0]),
        "c2": np.array([0.0, 1.0]),
        "c3": np.array([-1.0, 0.0]),
    }

    result = retrieve_text_only(_example(), _chunks(), embeddings, np.array([1.0, 0.0]), top_k=2)

    assert result.condition == "text"
    assert result.reader_id is None
    assert result.ranked_chunks[0][0] == "c1"


def test_retrieve_gaze_view_accepts_required_conditions() -> None:
    embeddings = {
        "c1": np.array([1.0, 0.0]),
        "c2": np.array([0.0, 1.0]),
        "c3": np.array([-1.0, 0.0]),
    }

    result = retrieve_gaze_view(
        _example(),
        "r1",
        _chunks(),
        embeddings,
        np.array([0.0, 1.0]),
        top_k=1,
        condition="actual_gaze",
    )

    assert result.condition == "actual_gaze"
    assert result.reader_id == "r1"
    assert result.ranked_chunks == [("c2", 1.0)]


def test_hybrid_retrieval_alpha_controls_text_gaze_mix() -> None:
    query = np.array([1.0, 0.0])
    text_rows = [
        (
            "text:c1",
            np.array([1.0, 0.0]),
            {"chunk_id": "c1", "text": "alpha", "char_start": 0, "char_end": 5},
        ),
        (
            "text:c2",
            np.array([0.0, 1.0]),
            {"chunk_id": "c2", "text": "beta", "char_start": 6, "char_end": 10},
        ),
    ]
    gaze_rows = [
        (
            "gaze:c1",
            np.array([0.0, 1.0]),
            {"chunk_id": "c1", "text": "alpha", "char_start": 0, "char_end": 5},
        ),
        (
            "gaze:c2",
            np.array([1.0, 0.0]),
            {"chunk_id": "c2", "text": "beta", "char_start": 6, "char_end": 10},
        ),
    ]

    text_heavy = _retrieve_hybrid_group("q1", "r1", 0.75, text_rows, gaze_rows, query, top_k=2)
    gaze_heavy = _retrieve_hybrid_group("q1", "r1", 0.25, text_rows, gaze_rows, query, top_k=2)

    assert text_heavy.condition == "hybrid_gaze_alpha_0p75"
    assert gaze_heavy.condition == "hybrid_gaze_alpha_0p25"
    assert text_heavy.ranked_chunks[0][0] == "c1"
    assert gaze_heavy.ranked_chunks[0][0] == "c2"
    assert text_heavy.metadata["evidence"][0]["alpha"] == 0.75


def test_text_candidate_gaze_rerank_filters_to_text_candidates() -> None:
    query = np.array([1.0, 0.0])
    text_rows = [
        (
            "text:c1",
            np.array([1.0, 0.0]),
            {"chunk_id": "c1", "text": "alpha", "char_start": 0, "char_end": 5},
        ),
        (
            "text:c2",
            np.array([0.8, 0.6]),
            {"chunk_id": "c2", "text": "beta", "char_start": 6, "char_end": 10},
        ),
        (
            "text:c3",
            np.array([0.0, 1.0]),
            {"chunk_id": "c3", "text": "gamma", "char_start": 11, "char_end": 16},
        ),
    ]
    gaze_rows = [
        (
            "gaze:c1",
            np.array([0.0, 1.0]),
            {"chunk_id": "c1", "text": "alpha", "char_start": 0, "char_end": 5},
        ),
        (
            "gaze:c2",
            np.array([1.0, 0.0]),
            {"chunk_id": "c2", "text": "beta", "char_start": 6, "char_end": 10},
        ),
        (
            "gaze:c3",
            np.array([1.0, 0.0]),
            {"chunk_id": "c3", "text": "gamma", "char_start": 11, "char_end": 16},
        ),
    ]

    result = _retrieve_text_candidate_gaze_rerank_group(
        "q1",
        "r1",
        candidate_top_n=2,
        text_rows=text_rows,
        gaze_rows=gaze_rows,
        query=query,
        top_k=1,
    )

    assert result.condition == "text_top2_gaze_rerank"
    assert result.ranked_chunks == [("c2", 1.0)]
    assert result.metadata["candidate_top_n"] == 2
    assert result.metadata["evidence"][0]["text_candidate_rank"] == 2
