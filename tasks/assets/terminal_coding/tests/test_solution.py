"""Hidden test suite. Injected into the verifier sandbox only — the agent
never sees this file. Tests the submission's solution module."""

import random

import solution


def test_basic_roundtrip():
    for s in ["", "a", "aaa", "aaabbbcccd", "wwwwaaadexxxxxx"]:
        assert solution.rle_decode(solution.rle_encode(s)) == s


def test_known_encoding():
    assert solution.rle_encode("aaabbbcccd") == "a3b3c3d1"
    assert solution.rle_encode("") == ""
    assert solution.rle_encode("x") == "x1"


def test_known_decoding():
    assert solution.rle_decode("a3b3c3d1") == "aaabbbcccd"
    assert solution.rle_decode("") == ""


def test_random_roundtrip():
    alphabet = "ab"
    rng = random.Random(0)
    for _ in range(50):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        assert solution.rle_decode(solution.rle_encode(s)) == s


def test_multidigit_runs():
    s = "a" * 12 + "b" * 3
    assert solution.rle_encode(s) == "a12b3"
    assert solution.rle_decode("a12b3") == s
