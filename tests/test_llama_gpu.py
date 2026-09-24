import math
import os

import pytest

pytestmark = pytest.mark.gpu

MODEL = os.environ.get("KODDS_MODEL")


@pytest.fixture(scope="module")
def backend():
    if not MODEL:
        pytest.skip("KODDS_MODEL not set")
    from kodds.llama import LlamaBackend

    return LlamaBackend(MODEL)


@pytest.fixture(scope="module")
def scorer(backend):
    from kodds.scorer import Scorer

    return Scorer(backend)


def test_obvious_answer_wins(scorer):
    probs = scorer.score(
        "What is the capital of France? Answer with the city name only.",
        ["Paris", "London", "Berlin"],
    )
    assert math.isclose(sum(probs.values()), 1.0)
    assert probs["Paris"] > 0.9


def test_multi_token_choices_sharing_a_prefix(scorer):
    probs = scorer.score(
        "Which planet is known as the Red Planet? Answer with the name only.",
        ["Mars", "Mars bar", "Mercury"],
    )
    assert probs["Mars"] > probs["Mars bar"]
    assert probs["Mars"] > probs["Mercury"]


def test_prefix_reuse_matches_a_cold_evaluation(backend, scorer):
    prompt, choices = "Is water wet? Answer yes or no.", ["yes", "no"]
    warm = scorer.logprobs(prompt, choices)
    cold = {}
    for choice in choices:
        backend.llm.reset()  # force the uncached path for each choice
        cold.update(scorer.logprobs(prompt, [choice]))
    for choice in choices:
        assert math.isclose(warm[choice], cold[choice], abs_tol=1e-2)
