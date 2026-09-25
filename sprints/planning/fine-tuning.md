# Fine-tuning plan

> Whether, when and how to post-train the scoring model. Tentative: nothing
> here is scheduled until the gate below says it is worth doing.

## Why kodds suits it

kodds reads exactly one quantity — the summed log-prob of a choice's tokens
plus `<|im_end|>`, normalised over the choices. Supervised fine-tuning's
loss is the cross-entropy of those same answer tokens, so training pushes
directly on the number kodds reports. (Tool-calling fine-tunes, by contrast,
teach a format and a habit and help only indirectly.)

## The gate — do this only if

Calibration (002) and prompt work have been tried first, because both are
cheap and neither bakes anything into weights. Fine-tune a task when, after
them:

- its accuracy on held-out items is still below what its consumer needs
  (routing is the only candidate today: 47% title-only on the 14B in 001);
- there is enough **real** labelled data to train on without touching the
  eval split — hundreds of items at minimum, thousands preferred;
- a consumer is waiting on it, so the gain has somewhere to go.

Severity (91%) and triage (98%) do not need it. Their gap is calibration,
not accuracy.

## Method

Cheapest first; each is a LoRA adapter on the default base, never a full
fine-tune.

1. **Plain SFT.** (prompt, correct choice) pairs, loss masked to the answer
   tokens. Prompts rendered **byte-for-byte as `Scorer` renders them** —
   same chat template, the empty `<think>\n\n</think>\n\n` block, the
   `<|im_end|>` terminator. Train/score format skew means tuning a
   distribution kodds never reads.
2. **Choice-normalised loss** (the better fit). Score every choice as
   `Scorer` does, softmax over the choices only, loss = −log p(correct).
   This trains the exact distribution kodds returns and pushes mass *off*
   the wrong choices, which plain SFT does not. Costs a custom training loop
   (one forward pass per choice, sharing the prompt's KV as `Scorer`
   already does).
3. **Distillation** — soft labels from a stronger model (Claude, or a large
   model on kai). For tasks with no ground truth. Imports the teacher's
   biases along with its skill.

DPO / RL are not worth it here: with no generation, a "preference" is just
a classification label.

## Data

- **Routing — korg history.** Every work item is already filed under its
  project: thousands of real labels (title + content), free. The obvious
  first training set.
- **Severity** — the 001 labels are hand-made against kmon's rubric. Real
  training data needs kmon to log the calls it makes (and their outcomes).
- **Triage** — synthetic today; no real source identified.
- **Splits are fixed before training.** The eval split never trains, and
  temperature/calibration is fitted on a third split, not on either.

## Don't bake in the label set

Projects come and go (kpidash is already retired). An adapter that
memorises "these eight names" breaks the day the set changes; one that
learned to *read the routing contracts* transfers. So:

- keep the project descriptions in every training prompt;
- shuffle their order, and train on varying subsets of the project list;
- hold out whole projects from training and measure on them — that number,
  not in-distribution accuracy, says whether the adapter generalises.

## Pipeline

Training runs on the Hugging Face safetensors base (`peft` or unsloth,
QLoRA), not on the GGUF. Then either:

- convert the adapter with llama.cpp's `convert_lora_to_gguf.py` and load it
  with llama-cpp-python's `lora_path` — one adapter per task over a shared
  resident base; or
- merge, convert and requantize to Q4_K_M.

Re-run the bake-off on the quantized result: quantization shifts the very
probabilities that were tuned.

## Calibration after training

Fine-tuning on small data makes models *more* confident, and the raw model
already needs T≈10. Re-fit the 002 calibration on the held-out calibration
split after every adapter; never ship an adapter's raw probabilities.

## Hardware — decided

Decided by Ken, 2026-09-24: **if training happens, it runs on kai's 5090,
and the adapter targets the default 14B** (Qwen3-14B Q4_K_M). No 8B
fallback model just to fit training on kubs0.

- Training on kai: pull the safetensors base there, train QLoRA, convert
  the adapter to GGUF, and copy it to kubs0. kai's 5090 also runs the RA's
  kvllm, so schedule the training run around it rather than evicting it.
- Inference stays on kubs0 beside klams' TEI routers. A LoRA adapter adds
  little VRAM on top of the 11.6 GiB peak measured in 001; re-measure it
  with the adapter loaded before shipping.

## Open decisions

- Whether routing's consumer (korg) wants kodds at all, before any training
  investment.
