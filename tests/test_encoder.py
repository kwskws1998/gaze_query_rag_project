import numpy as np

from gaze_query_rag.modeling.encoder import build_word_to_token_alignment
from gaze_query_rag.modeling.gaze_features import word_trt_to_token_trt


def test_build_word_to_token_alignment_handles_subtokens() -> None:
    text = "Leading water scientists."
    words = ["Leading", "water", "scientists."]
    offsets = [(-1, -1), (0, 4), (4, 7), (8, 13), (14, 24), (24, 25)]

    alignment = build_word_to_token_alignment(text, words, offsets)

    assert alignment == [[1, 2], [3], [4, 5]]


def test_build_word_to_token_alignment_handles_attached_quote() -> None:
    text = "“There will not be enough water.”"
    words = ["There", "will", "not", "be", "enough", "water."]
    offsets = [(-1, -1), (1, 6), (7, 11), (12, 15), (16, 18), (19, 25), (26, 32)]

    alignment = build_word_to_token_alignment(text, words, offsets)

    assert alignment == [[1], [2], [3], [4], [5], [6]]


def test_word_to_token_alignment_preserves_trt_mass_after_mapping() -> None:
    text = "Leading water"
    words = ["Leading", "water"]
    offsets = [(0, 4), (4, 7), (8, 13)]
    alignment = build_word_to_token_alignment(text, words, offsets)

    token_trt = word_trt_to_token_trt(np.array([10.0, 5.0]), alignment, num_tokens=3)

    assert np.allclose(token_trt, np.array([5.0, 5.0, 5.0]))


def test_build_word_to_token_alignment_handles_hyphen_split_interest_area() -> None:
    text = "Unlike mopeds, e-bicycles are usually permitted."
    words = ["Unlike", "mopeds,", "e-", "bicycles", "are", "usually", "permitted."]
    offsets = [(0, 6), (7, 14), (15, 17), (17, 25), (26, 29), (30, 37), (38, 47), (47, 48)]

    alignment = build_word_to_token_alignment(text, words, offsets)

    assert alignment == [[0], [1], [2], [3], [4], [5], [6, 7]]
