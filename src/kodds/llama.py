"""The llama-cpp-python backend: a GGUF on the GPU, prompt prefix reused.

The prompt is evaluated once; each choice then truncates the KV cache back to
the end of the prompt and evaluates only its own tokens. ``Llama.eval`` drops
KV entries from ``n_tokens`` onward before it runs, so rewinding ``n_tokens``
is the whole of the cache management.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from llama_cpp import Llama

from kodds.scorer import Scorer


class LlamaBackend:
    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 1024,
        n_gpu_layers: int = -1,
        verbose: bool = False,
    ):
        # logits_all: every evaluated position keeps its logits, which is
        # what lets one eval of a multi-token choice score all its tokens.
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            logits_all=True,
            verbose=verbose,
        )

    def tokenize(self, text: str) -> list[int]:
        return self.llm.tokenize(text.encode(), add_bos=False, special=True)

    def continuation_logits(
        self, prefix: Sequence[int], continuation: Sequence[int]
    ) -> np.ndarray:
        if not prefix:
            raise ValueError("need a non-empty prefix to predict from")
        n = len(prefix)
        if n + len(continuation) > self.llm.n_ctx():
            raise ValueError(
                f"{n + len(continuation)} tokens exceed n_ctx={self.llm.n_ctx()}"
            )
        cached = self.llm.n_tokens >= n and list(self.llm.input_ids[:n]) == list(prefix)
        if cached:
            self.llm.n_tokens = n
        else:
            self.llm.reset()
            self.llm.eval(prefix)
        self.llm.eval(continuation)
        # Row n-1 is the prediction after the prompt; each later row after one
        # more choice token. The last choice token's own row predicts nothing
        # we score.
        return np.array(self.llm.scores[n - 1 : n - 1 + len(continuation)])


def load(model_path: str | Path, **kwargs) -> Scorer:
    """A ready ``Scorer`` over a Qwen3 GGUF."""
    return Scorer(LlamaBackend(model_path, **kwargs))
