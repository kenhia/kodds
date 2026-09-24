import math

import pytest

from kodds import normalize


def test_sums_to_one():
    probs = normalize({"Spam": -1.0, "Legitimate": -2.0, "Phishing": -3.0})
    assert math.isclose(sum(probs.values()), 1.0)
    assert probs["Spam"] > probs["Legitimate"] > probs["Phishing"]


def test_survives_very_negative_logprobs():
    probs = normalize({"a": -1000.0, "b": -1001.0})
    assert math.isclose(probs["a"], 1 / (1 + math.exp(-1)))


def test_rejects_empty():
    with pytest.raises(ValueError):
        normalize({})
