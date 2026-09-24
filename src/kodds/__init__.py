"""kodds — per-choice probabilities from a local model's logits."""

import math


def normalize(logprobs: dict[str, float]) -> dict[str, float]:
    """Turn per-choice log-probabilities into probabilities summing to 1.

    Done in log space (log-sum-exp) so very negative scores don't underflow.
    """
    if not logprobs:
        raise ValueError("need at least one choice")
    peak = max(logprobs.values())
    total = peak + math.log(sum(math.exp(v - peak) for v in logprobs.values()))
    return {choice: math.exp(v - total) for choice, v in logprobs.items()}
