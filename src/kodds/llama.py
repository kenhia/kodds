"""The llama-cpp-python backend: a GGUF on the GPU, prompt prefix reused.

The prompt is evaluated once; each choice then truncates the KV cache back to
the end of the prompt and evaluates only its own tokens. Across calls, the
longest run of tokens shared with the previous prompt is kept too, so a task's
fixed instructions and choice list are evaluated once, not once per input.
``Llama.eval`` drops KV entries from ``n_tokens`` onward before it runs, so rewinding ``n_tokens``
is the whole of the cache management.
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from llama_cpp import Llama

from kodds.scorer import Scorer

PREFIX_BLOCK = 128


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
        # Keep the longest cached run that matches the new prefix, and evaluate
        # only what follows it: a task's instructions and choice list come
        # before its input, so across calls only the input is new.
        #
        # The prefix is evaluated in fixed blocks aligned to absolute
        # positions, and reuse stops at a block boundary. llama.cpp's results
        # depend on how tokens are split into batches (measured: up to ~1 nat
        # on one logit), so without this a prompt's probabilities would depend
        # on whatever was scored before it. With it, a prompt is always
        # computed the same way, cached or cold.
        keep = self._common_prefix(prefix)
        if keep < n:
            keep -= keep % PREFIX_BLOCK
        self.llm.n_tokens = keep
        for start in range(keep, n, PREFIX_BLOCK):
            self.llm.eval(prefix[start : start + PREFIX_BLOCK])
        self.llm.eval(continuation)
        # Row n-1 is the prediction after the prompt; each later row after one
        # more choice token. The last choice token's own row predicts nothing
        # we score.
        return np.array(self.llm.scores[n - 1 : n - 1 + len(continuation)])

    def _common_prefix(self, tokens: Sequence[int]) -> int:
        cached = self.llm.input_ids[: min(self.llm.n_tokens, len(tokens))]
        mismatch = np.flatnonzero(cached != np.asarray(tokens[: len(cached)]))
        return int(mismatch[0]) if mismatch.size else len(cached)


def load(model_path: str | Path, **kwargs) -> Scorer:
    """A ready ``Scorer`` over a Qwen3 GGUF, identified by its file name."""
    return Scorer(LlamaBackend(model_path, **kwargs), model=Path(model_path).name)
