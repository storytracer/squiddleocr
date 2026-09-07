import pytest

from squiddleocr.convert.codec import codec_to_dictionary, read_dict_file, write_dict_file
from squiddleocr.recognizers.ctc import ctc_greedy_decode


def test_dictionary_is_in_label_order(toy_c2l):
    assert codec_to_dictionary(toy_c2l) == [" ", "b", "a", "̈", "ſ"]


def test_dictionary_roundtrip_file(tmp_path, toy_c2l):
    chars = codec_to_dictionary(toy_c2l)
    p = tmp_path / "dict.txt"
    write_dict_file(chars, p)
    assert p.read_bytes() == " \nb\na\n̈\nſ\n".encode("utf-8")
    assert read_dict_file(p) == chars


@pytest.mark.parametrize(
    "c2l, msg",
    [
        ({"a": [1, 2]}, "one-to-one"),
        ({"a": [0]}, "reserved"),
        ({"a": [1], "b": [1]}, "used by both"),
        ({"a": [1], "b": [3]}, "contiguous"),
        ({"a\n": [1]}, "line break"),
        ({}, "Empty"),
    ],
)
def test_dictionary_rejects_bad_codecs(c2l, msg):
    with pytest.raises(ValueError, match=msg):
        codec_to_dictionary(c2l)


def test_greedy_decode_collapses_repeats_and_blanks():
    import numpy as np

    chars = ["a", "b"]
    # classes: 0 blank, 1 a, 2 b ; sequence a a _ a b b
    idx = [1, 1, 0, 1, 2, 2]
    probs = np.full((6, 3), 0.1)
    for t, i in enumerate(idx):
        probs[t, i] = 0.8
    text, score = ctc_greedy_decode(probs, chars)
    assert text == "aab"
    assert score == pytest.approx(0.8)
    assert ctc_greedy_decode(np.eye(3)[[0, 0]], chars) == ("", 0.0)


def test_real_codec_matches_network(medium_model):
    chars = codec_to_dictionary(medium_model.c2l)
    assert len(chars) + 1 == medium_model.num_classes
    assert all(len(c) == 1 for c in chars)
    assert chars[0] == " "  # space is label 1 in the kraken models
