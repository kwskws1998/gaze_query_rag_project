import numpy as np

from gaze_query_rag.retrieval.index import build_in_memory_index, search_index
from gaze_query_rag.retrieval.retriever import retrieve_gaze_view, retrieve_text_only
from gaze_query_rag.schemas import Chunk, QAExample


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
