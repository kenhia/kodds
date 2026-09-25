# 002 — calibration on the default model

korg: proposal 3196 — #3192 eval data + splits, #3193 routing prompt,
#3194 calibration methods, #3195 task abstraction.

## Goal

Make Qwen3-14B Q4_K_M's probabilities numbers you can threshold on, not just
a ranking. 001 measured ~0.96 mean confidence against 75% accuracy, and one
global T≈10 recovered about a third of the Brier score.

## Premise check (sprint start)

All four held. The eval set was 154 items (64/45/45), title-only routing was
47%, no task abstraction existed, and kubs0's GPU showed 2,883 MiB, which is
just klams' two TEI routers. One drift: #3192 calls kpidash retired, but
it is an active korg project (it was the kpidash *MCP tool* that was
retired), so it stays in the choice set.

Ken's planning edits (`sprints/planning/roadmap.md`, new
`sprints/planning/fine-tuning.md`) were uncommitted on `main` at the start.
He chose to have them ride along with this sprint.

## Eval data and splits (#3192)

**Decision (Ken): no korg content in the public repo.** kenhia/kodds is
public, and routing is now scored on real work items, title and content,
which describe a private homelab. Only the split *assignment* is committed:
`evals/splits/route.json`, one `[wi_number, project, split]` per item.
`evals/build_evalset.py` fetches the text from korg's REST API
(`kubsdb:5674`) into git-ignored `.scratch/evalset/`. `evals/README.md`
says the full corpus is available on request.

- **route** — the choice set is korg's 46 active projects with their
  one-line routing contracts (`eval` excluded: its own contract says
  it never holds real work). Every unarchived item with content in those
  projects was eligible: 1,611 items, and 44 of the 46 projects have at least
  one (`ATV` and `gratch` have none). The split is seeded (2026) and per
  project: min(12, 40%) of its items to cal, as many to test, the rest to
  train.

  | split | items | projects with ≥1 |
  |---|---|---|
  | cal | 300 | 39 |
  | test | 300 | 39 |
  | train | 1,011 | 44 |

  Per project (cal/test/train): k-homelab 12/12/219, korg 12/12/215,
  klams 12/12/106, agent-skills 12/12/72, kmon 12/12/59,
  kdeskdash 12/12/49, kaed 12/12/43, kfdc 12/12/30, kyac 12/12/23,
  karc 12/12/22, kdashdata 12/12/20, kvllm 12/12/16, claude-cleo 12/12/15,
  krot 12/12/15, kagviz 12/12/12, homelab-ai 12/12/8, kpidash 12/12/7,
  homelab-health 12/12/6, hv-simulator 10/10/7, kstudiodash 8/8/5,
  kctrldeck 8/8/4, kprojects 7/7/5, kapollo 5/5/4, kmuster 5/5/4,
  klaude-top 5/5/3, kwebi 4/4/4, knarr 4/4/4, klams-view 4/4/3,
  apt-temps 3/3/3, klams-mind 3/3/3, khlenv 3/3/2, kpolice 3/3/2,
  gh-kenhia 2/2/3, kodds 2/2/3, krcmd 2/2/2, kbrickshoot 2/2/2,
  mortars 2/2/1, kpidashclient-win 1/1/2, kfo 1/1/2, and agent-wiki,
  korg-dash, agent-projects, ktail and cross-project-planning train only.

  Labels are the project each item is filed under *today*, which carries
  some history: some older korg items predate kfdc and kfo, and today they
  would route there. Treat routing accuracy as a floor on that account.
- **severity** / **triage** — 001's 45 items each, split per label
  (seeded): 7 to cal, 8 to test → 21 cal / 24 test. **Severity labels are
  still hand-made** against kmon's rubric; kmon doesn't log its own calls, so
  there is no real ground truth yet. Triage is synthetic.
- Rows are `{task, id, split, inputs, label}`: inputs rather than rendered
  prompts, so the same row can be rendered under any prompt variant.
- 001's `evals/evalset.jsonl` and `bakeoff.py` stay as they were, as 001's
  frozen record.

## Task abstraction (#3195)

A named task is now the unit kodds serves (`kodds.tasks`). This is what
003's service will expose:

- `tasks/<name>.toml` is hand-written: `template` (a `str.format` text over the
  task's named inputs, plus `{choice_list}`), an optional `system`,
  per-input character `limits`, and `[choices]` as name → description, in
  order. severity and triage render **byte-identical** to 001's prompts
  (checked against `evalset.jsonl`), so their numbers carry over. route's
  `[choices]` table is generated from korg's active projects by
  `build_evalset.py projects`.
- `tasks/<name>.calibration.json` is machine-written by
  `evals/calibrate.py --write`, keyed by model file name. Each entry holds
  `{method, temperature, bias, choices, provenance}`, and one form covers
  every method: `softmax((logprob + bias[choice]) / T)`. Provenance records
  the prompt variant, the fit split and n, the date, and the test metrics
  both calibrated and raw.
- `task.classify(scorer, **inputs)` returns a `Result`: `probs`, `raw`,
  `logprobs`, `calibrated`, `model`. **The fallback:** if the loaded model
  has no fit, or the task's choice list differs from the one the fit
  recorded (which happens whenever a korg project is added or retired),
  `probs` is the raw distribution and `calibrated=False`. Another model's
  or another list's parameters are never borrowed.
- `Scorer` gains a `model` identity (`kodds.llama.load` sets it to the GGUF
  file name) and an optional `system` argument. `score(prompt, choices)`
  is unchanged.
- `task.content_free()` renders the prompt with every input set to `N/A`,
  which is the input contextual calibration needs.
- Tests (`tests/test_tasks.py`, model-free, on 001's bigram fake): rendering
  and truncation, missing/unknown inputs, bias-then-T maths, calibrated
  vs. raw, both fallbacks, system passthrough, TOML + JSON loading, and
  that the repo's own tasks load.

### Deterministic prompt caching (came out of #3193's cost)

Routing's prompt is now ~2k tokens of project list before the item, so
`LlamaBackend` keeps the longest cached run shared with the previous prompt
instead of resetting whenever the whole prompt differs. The GPU test for
this failed at first, and that exposed something worth recording:
**llama.cpp's logits depend on how the tokens are split into batches.**
The same prompt evaluated in one batch or in two differed by up to ~1 nat
on a logit (0.08 vs 0.07 in a two-choice probability). With naive reuse, an
item's probability would have depended on whichever item was scored before
it. The fix is to evaluate the prompt in fixed 128-token blocks aligned to
absolute positions, and to stop reuse at a block boundary. A prompt is then
computed identically cached or cold, and the GPU tests assert agreement to
1e-6.

## Routing prompt (#3193)

Measured on route's **cal** split (300 items, 46 choices; raw probabilities,
Qwen3-14B Q4_K_M, `n_ctx=8192`) with `evals/sweep.py`. Test was scored once
afterwards, for the variant that won.

| variant | acc | Brier | prompt tokens (mean) | p50 / p95 ms |
|---|---|---|---|---|
| title + one-line contracts (001's prompt) | 0.320 | 1.244 | 1,908 | 1,038 / 1,064 |
| + WI content (≤1,500 chars) | 0.467 | 0.952 | 2,186 | 1,204 / 1,275 |
| + WI content (≤500 chars) | 0.423 | 1.033 | 2,034 | 1,099 / 1,144 |
| + content, project list reversed | 0.447 | 1.008 | 2,186 | 1,209 / 1,280 |
| + content, 16 few-shot demonstrations (title → project) | 0.433 | 1.074 | 2,684 | 1,220 / 1,293 |
| + content, notes excerpt (≤400 chars) per project | 0.590 | 0.775 | 6,255 | 1,382 / 1,486 |
| + content, 3 example titles per project | 0.630 | 0.683 | 5,125 | 1,341 / 1,439 |
| + content, 5 example titles per project | 0.650 | 0.635 | 6,376 | 1,383 / 1,483 |
| **+ content, notes excerpt (≤200) + 3 example titles** | **0.670** | **0.600** | 7,270 | 1,406 / 1,520 |

Chosen: the last row. Reading it:

- **The WI's content is worth 15 points** over its title, and cutting
  it to 500 characters gives a third of that back.
- **Richer contracts beat demonstrations.** Three real titles under each
  project's contract (+16 points) do far more than 16 classic few-shot
  demonstrations, which *hurt* (−3). With 46 choices, 16 random demos
  covered only 7 projects, and the model copies them: 189 of 300 picks landed
  on those 7 labels (104 without demos; 84 items truly belong there).
  Examples attached to every contract teach each project's vocabulary.
- **Order is part of the prompt.** Reversing the project list changed
  116 of 300 picks (39%) and cost 2 points. Picks shift towards wherever the
  attractor projects sit (below), so position bias and label bias can't be
  separated here. Either way, a calibration is only valid for the order it
  was fitted on, and the prompt hash (below) pins that.
- **Cost.** The prompt grows to ~7.3k tokens, but the project list is the
  same on every call and is cached (deterministically, see #3195's section),
  so each call pays for the item and the choices only. p50 goes from 1.0 s to
  1.4 s. **Nearly all of that is 46 per-choice evaluations**, not the
  prompt. The roadmap's "batch a call's choices" item, parked as
  latency-only at p50 95 ms, is now route's main cost. Peak GPU was
  13.4 GiB total (10.5 GiB this process, n_ctx 8192) beside TEI, and both TEI
  pids were untouched on every run.
- Full project notes (~57k characters, ~15k tokens, nine projects with
  none) don't fit beside TEI, so notes were tested as capped excerpts.

**Ken's question on #3193: would a multi-call, coarse-to-fine routing do
better?** The error pattern says it would not help much. On the
content variant, the misses are dominated by **attractor labels**, not by
confusion between sibling projects. `kprojects` was picked 49 times against 7
true items, and `k-homelab` (30), `korg` (21) and `krot` (15) pull similarly.
The one large semantic confusion is homelab-health → k-homelab (9 of 12:
findings about a host read as changes to it). A first-pass category call
would inherit the same attractors. A per-choice bias correction targets
them directly (#3194), and richer contracts removed most of them anyway.
Hierarchy stays an idea for a routing set that grows much beyond 46.

**Decision (Ken): the enriched descriptions ship as a private overlay.**
The winning text embeds real WI titles and korg-notes excerpts (hosts,
paths, the tailnet domain), and kodds is public. `tasks/route.toml` keeps the
template and the one-line contracts. `build_evalset.py overlay` generates
`tasks/private/route.json` (git-ignored) on the serving host. Every
calibration now pins `prompt_sha256`, a hash of the rendered content-free
prompt plus system prompt. Without the overlay, or with one that has drifted
(new notes, a new title sampled), the hash differs and route serves raw
probabilities with `calibrated=False`. `calibrate.py --write` refuses to
save a fit unless the swept variant hashes the same as the task as it will
be served. The overlay generated for this sprint renders byte-identical to
the variant scored on cal.

## Calibration methods (#3194)

`evals/calibrate.py` reads the sweep's log-prob dumps (committed under
`evals/results/002/`: ids, labels and log-probs only, no text), so it needs
no GPU. Every method has the form `softmax((logprob + bias) / T)`: raw,
one global T pooled over every task (001's baseline), T per task,
contextual calibration (bias = −content-free log-prob) with and without T,
and a per-choice bias with T at two shrinkage strengths. Methods are fitted
on cal and reported on test.

**Route** (test n=300, the shipped prompt):

| method | T | CV NLL / acc (cal) | test acc | Brier | NLL | ECE | mean conf |
|---|---|---|---|---|---|---|---|
| raw | 1.00 | 5.138 / 0.670 | 0.610 | 0.707 | 6.075 | 0.336 | 0.95 |
| global-T | 5.05 | 1.589 / 0.670 | 0.610 | 0.558 | 1.773 | 0.071 | 0.61 |
| task-T | 5.09 | 1.589 / 0.670 | 0.610 | 0.559 | 1.772 | 0.074 | 0.61 |
| cc | 1.00 | 5.770 / 0.657 | 0.590 | 0.740 | 6.766 | 0.356 | 0.95 |
| cc+T | 5.28 | 1.777 / 0.657 | 0.590 | 0.588 | 1.957 | 0.067 | 0.58 |
| bias+T | 5.05 | 1.575 / 0.670 | 0.607 | 0.555 | 1.758 | 0.087 | 0.61 |
| **bias+T/light** | 4.44 | 1.376 / 0.687 | **0.647** | **0.528** | **1.548** | 0.071 | 0.69 |

Test reliability for the chosen method (confidence bin → n, mean
confidence, accuracy): 0.1–0.2 13 / 0.17 / 0.08 · 0.2–0.3 17 / 0.24 /
0.29 · 0.3–0.4 23 / 0.35 / 0.39 · 0.4–0.5 25 / 0.45 / 0.52 · 0.5–0.6 18 /
0.54 / 0.56 · 0.6–0.7 24 / 0.66 / 0.58 · 0.7–0.8 41 / 0.76 / 0.76 · 0.8–0.9
62 / 0.85 / 0.82 · **0.9–1.0 77 / 0.94 / 0.78**. The curve follows the
diagonal except in the top bin, which is still overconfident.

- **Temperature does most of the work.** T≈5 takes NLL from 6.1 to 1.8
  and ECE from 0.34 to 0.07 without moving a single pick. The fitted
  per-choice bias adds +3.7 points of accuracy and the rest of the Brier
  gain by pushing back on attractor projects: kprojects −6.5 and korg −2.9
  against a median of +0.9, on the pre-T scale.
- **Contextual calibration helped the plain prompt and hurt the rich
  one.** On the contracts-only prompt, subtracting the content-free
  log-probs lifted cal accuracy 46.7% → 51.7% with no labels at all. Once
  the contracts carry example titles, the attractors are mostly gone, and
  CC over-corrects (−2 points; cc+T is worse than T alone). It stays
  in the script. It is the one method that could be refitted without
  labels when the project list changes.
- **Severity / triage** (test n=24 each): both keep the pooled **global T**
  (5.05). Severity: raw 95.8% / Brier 0.022 / NLL 0.034 → 95.8% / 0.055 /
  0.130. Triage: raw 95.8% / 0.083 / 0.339 → 95.8% / 0.053 / 0.093. On
  severity's test split the raw model was already well calibrated and T
  makes it underconfident. On its cal split the opposite held (three
  confident misses in 21). **Neither split is big enough to settle this.**
  The honest statement is that these two need real labelled traffic (kmon
  logging its calls: 004), not a better method.

**How the method is chosen**, rules that each came from a measured failure:

1. **T ≥ 1.** Triage's cal split is perfectly separable (21/21), so the
   likelihood drove T to 0.05, sharpening a model that was already certain.
   Its one test miss then cost 6.8 nats.
2. **global-T is the default, and a task earns its own method only by
   beating it by more than one standard error** of the paired per-item NLL
   under 5-fold CV within cal. With 21 cal items, the lowest CV NLL out of
   seven methods is noise. (It had picked cc+T for triage and a free bias
   for severity.)
3. **A task with no cal errors keeps the default.** Without errors, any
   T > 1 costs a little on every item, so "sharper" wins consistently and
   means nothing.
4. **A choice with no cal items keeps the median bias.** Unmasked, the
   seven projects cal never shows (gratch, agent-wiki, korg-dash, ktail,
   agent-projects, cross-project-planning, ATV) were fitted to ≈ −25, a
   probability factor of about e⁻⁶, which made them unroutable. The masked
   fit gives up 0.05 NLL on test. Test holds none of those seven either, so
   the neutral bias only leaks them mass there. On live traffic, that is
   the correct trade.
5. Shrinkage: `bias+T` (L2 0.01) was too strong. `bias+T/light` (L2 1e-4)
   matched the unshrunk fit in CV (1.379 vs 1.387) and tamed the bias range
   (−7.5..15.7 vs −11.7..18.5).

`calibrate.py --write <model>` writes each task's chosen fit to
`tasks/<task>.calibration.json`, with provenance (fit split and n, date,
prompt variant, CV NLL, test metrics calibrated and raw) and the prompt
hash. It refuses if the swept variant doesn't hash the same as the task as
it will be served.

**End to end (GPU):** `tasks["route"].classify(scorer, …)` on a test item
returned calibrated probabilities, and its log-probs matched the sweep's to
1e-6 from a cold cache. The overlay's content-free log-probs re-scored
identically to the cal run's, which confirms the cal dumps and the shipped
prompt are the same thing.

## For the fine-tuning gate (fine-tuning.md)

**Routing on the default 14B after prompt work + calibration, held-out test
(300 items, 46 projects): top-1 64.7%, top-3 80.7%, Brier 0.528, NLL 1.55,
ECE 0.071.** At confidence ≥ 0.5 it covers 74% of items at 74.8% accuracy.
At ≥ 0.8 it covers 46% at 79.9%, and accuracy plateaus near 80% however high
the threshold goes. That is enough to *suggest* a project (top 3, or a gated
"probably X"). It is not enough to auto-file without review, which would
want ~95% at useful coverage. Of the gate's three conditions: accuracy
short of an auto-route consumer, **yes**; enough real data to train on
without touching test, **yes** (1,011 train items, thousands in korg);
a consumer waiting, **no**. Whether korg wants kodds routing at all is
still fine-tuning.md's open decision. The gate stays shut until that is
answered.

## Follow-ups

- **korg #3197**: route's bias was fitted on a cap-balanced cal split, so
  it encodes a near-uniform prior. Whether to correct it to a consumer's
  real project mix is a decision for 004's consumer.
- 003: the serving host must generate route's overlay (noted on the
  roadmap), and the API should surface `calibrated` rather than hide it.
- Choice batching is now route's main latency (1.4 s p50; roadmap "Later",
  text updated).
- Severity/triage calibration needs real labelled traffic (004), not more
  method work.

## Repaired in passing

Nothing outside the covered items needed repair.
