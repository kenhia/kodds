import math

import numpy as np
import pytest

from kodds.scorer import Scorer, qwen3_prompt, sequence_logprob

EOT = 3  # "<|im_end|>" in the fake vocabulary
VOCAB = 128


class BigramBackend:
    """A fake model: one character per token, next token depends on the last.

    ``table[last][next]`` is a logit; everything unlisted is very unlikely.
    """

    def __init__(self, table: dict[str, dict[str, float]]):
        self.table = table

    def tokenize(self, text: str) -> list[int]:
        if text == "<|im_end|>":
            return [EOT]
        return [ord(c) for c in text]

    def _row(self, last: int) -> np.ndarray:
        row = np.full(VOCAB, -20.0)
        for nxt, logit in self.table.get(chr(last), {}).items():
            row[EOT if nxt == "$" else ord(nxt)] = logit
        return row

    def continuation_logits(self, prefix, continuation):
        seen = list(prefix) + list(continuation)
        start = len(prefix) - 1
        return np.array([self._row(seen[start + i]) for i in range(len(continuation))])


def plain(prompt: str) -> str:
    return prompt


def test_sequence_logprob_is_summed_log_softmax():
    rows = np.array([[0.0, math.log(3.0)], [math.log(1.0), math.log(1.0)]])
    # token 1 at p=3/4, then token 0 at p=1/2
    assert math.isclose(sequence_logprob(rows, [1, 0]), math.log(0.75 * 0.5))


def test_sequence_logprob_survives_huge_logits():
    rows = np.array([[1e4, 1e4 - 1.0]])
    assert math.isclose(sequence_logprob(rows, [0]), -math.log(1 + math.exp(-1)))


def test_sequence_logprob_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        sequence_logprob(np.zeros((2, 4)), [0])


def test_choices_sharing_a_first_token_are_told_apart():
    # After ":" the model is sure of "s"; after "s" it prefers "p" to "h".
    # A first-token reader would call "spam" vs "shop" a tie.
    model = BigramBackend(
        {
            ":": {"s": 5.0},
            "s": {"p": 3.0, "h": 0.0},
            "p": {"a": 5.0},
            "a": {"m": 5.0},
            "m": {"$": 5.0},
            "h": {"o": 5.0},
            "o": {"p": 5.0},
        }
    )
    probs = Scorer(model, template=plain).score("x:", ["spam", "shop"])
    assert probs["spam"] > 0.9
    assert math.isclose(sum(probs.values()), 1.0)


def test_end_of_turn_stops_a_prefix_choice_winning_for_free():
    # After "o" the model strongly continues to "k", not to end of turn: the
    # answer is "ok", and "o" alone is unlikely.
    model = BigramBackend(
        {":": {"o": 5.0}, "o": {"k": 5.0, "$": -5.0}, "k": {"$": 5.0}}
    )
    with_eot = Scorer(model, template=plain).score(":", ["o", "ok"])
    assert with_eot["ok"] > 0.99
    # Without the terminator, "o" contains every continuation of itself.
    without = Scorer(model, template=plain, end_of_turn=None).score(":", ["o", "ok"])
    assert without["o"] > without["ok"]


def test_rejects_empty_and_duplicate_choices():
    scorer = Scorer(BigramBackend({}), template=plain)
    with pytest.raises(ValueError):
        scorer.score("x", [])
    with pytest.raises(ValueError):
        scorer.score("x", ["a", "a"])


def test_qwen3_prompt_disables_thinking():
    text = qwen3_prompt("Is it spam?", system="Be brief.")
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert text.startswith("<|im_start|>system\nBe brief.<|im_end|>\n")
    assert "<|im_start|>user\nIs it spam?<|im_end|>\n" in text
