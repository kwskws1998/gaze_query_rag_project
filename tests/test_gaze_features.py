import numpy as np

from gaze_query_rag.modeling.gaze_features import (
    compute_gaze_distribution,
    compute_mean_gaze,
    shuffle_reader_gaze,
    word_trt_to_token_trt,
)


def test_gaze_distribution_sums_to_one() -> None:
    gaze = compute_gaze_distribution(np.array([0.0, 10.0, 30.0]))

    assert np.isclose(gaze.sum(), 1.0)
    assert np.all(gaze > 0)


def test_zero_trt_case_does_not_produce_nan() -> None:
    gaze = compute_gaze_distribution(np.zeros(4))

    assert not np.isnan(gaze).any()
    assert np.isclose(gaze.sum(), 1.0)
    assert np.allclose(gaze, np.full(4, 0.25))


def test_word_to_token_trt_mass_is_preserved() -> None:
    word_trt = np.array([10.0, 5.0, 0.0])
    mapping = [[0, 1], [2], []]
    token_trt = word_trt_to_token_trt(word_trt, mapping, num_tokens=4)

    assert np.isclose(token_trt.sum(), 15.0)
    assert np.allclose(token_trt, np.array([5.0, 5.0, 5.0, 0.0]))


def test_mean_gaze_normalizes_output() -> None:
    mean = compute_mean_gaze(
        {
            "r1": np.array([0.8, 0.2]),
            "r2": np.array([0.2, 0.8]),
        }
    )

    assert np.allclose(mean, np.array([0.5, 0.5]))


def test_shuffled_gaze_does_not_map_reader_to_self_when_multiple_readers_exist() -> None:
    gaze_by_reader = {
        "r1": np.array([1.0, 0.0, 0.0]),
        "r2": np.array([0.0, 1.0, 0.0]),
        "r3": np.array([0.0, 0.0, 1.0]),
    }

    shuffled = shuffle_reader_gaze(gaze_by_reader, seed=7)

    for reader_id, gaze in shuffled.items():
        assert not np.array_equal(gaze, gaze_by_reader[reader_id])
