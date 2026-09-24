"""score(prompt, choices): per-choice probabilities from one model's logits.

Each choice is scored as the SUM of its tokens' log-probs given the prompt,
followed by the end-of-turn token. The sum is what makes multi-token choices
and choices sharing a first token work; the end-of-turn token is what makes a
choice mean "the answer is exactly this" — without it, "Spam" would always
outscore "Spam folder", since every continuation of "Spam" counts for it.

The model sits behind the small ``Backend`` protocol so all of this is
testable without weights; ``kodds.llama.LlamaBackend`` is the real one.
"""

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from kodds import normalize


class Backend(Protocol):
    def tokenize(self, text: str) -> list[int]:
        """Tokens for ``text``, special tokens parsed, no BOS added."""
        ...

    def continuation_logits(
        self, prefix: Sequence[int], continuation: Sequence[int]
    ) -> np.ndarray:
        """Logits predicting each token of ``continuation`` after ``prefix``.

        Shape ``(len(continuation), n_vocab)``: row ``i`` is the next-token
        distribution (unnormalised) the model produces having seen
        ``prefix + continuation[:i]``.
        """
        ...


def qwen3_prompt(prompt: str, system: str | None = None) -> str:
    """Qwen3's chat template with thinking disabled, up to the answer.

    This is exactly what the template renders for ``enable_thinking=False``:
    an empty think block, so the next token is the answer rather than
    ``<think>``.
    """
    head = f"<|im_start|>system\n{system}<|im_end|>\n" if system else ""
    return (
        f"{head}<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


QWEN3_END_OF_TURN = "<|im_end|>"


def sequence_logprob(rows: np.ndarray, tokens: Sequence[int]) -> float:
    """Summed log-prob of ``tokens``, row ``i`` of ``rows`` predicting token ``i``.

    Log-softmax per row in float64 with the max subtracted, so large logits
    neither overflow nor lose the small differences that are the answer.
    """
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] != len(tokens):
        raise ValueError(f"{rows.shape} logit rows for {len(tokens)} tokens")
    peak = rows.max(axis=1, keepdims=True)
    log_z = peak[:, 0] + np.log(np.exp(rows - peak).sum(axis=1))
    picked = rows[np.arange(len(tokens)), list(tokens)]
    return float((picked - log_z).sum())


class Scorer:
    def __init__(
        self,
        backend: Backend,
        *,
        template=qwen3_prompt,
        end_of_turn: str | None = QWEN3_END_OF_TURN,
    ):
        self.backend = backend
        self.template = template
        self.end_of_turn = end_of_turn

    def logprobs(self, prompt: str, choices: Sequence[str]) -> dict[str, float]:
        """Summed log-prob of each choice (plus end-of-turn) given ``prompt``."""
        if not choices:
            raise ValueError("need at least one choice")
        if len(set(choices)) != len(choices):
            raise ValueError("choices must be distinct")
        prefix = self.backend.tokenize(self.template(prompt))
        tail = self.backend.tokenize(self.end_of_turn) if self.end_of_turn else []
        out = {}
        for choice in choices:
            tokens = self.backend.tokenize(choice) + tail
            rows = self.backend.continuation_logits(prefix, tokens)
            out[choice] = sequence_logprob(rows, tokens)
        return out

    def score(self, prompt: str, choices: Sequence[str]) -> dict[str, float]:
        """Probability of each choice, normalised over ``choices``."""
        return normalize(self.logprobs(prompt, choices))
