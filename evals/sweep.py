"""Score one task's split under prompt variants and dump the log-probs.

    uv run python evals/sweep.py ~/models/gguf/Qwen3-14B-Q4_K_M.gguf route cal \\
        title content content-examples

Each variant is the task from ``tasks/<task>.toml`` with a different template
or choice descriptions. Rows go to ``evals/results/002/<task>.<variant>.<split>.jsonl``
(ids, labels and log-probs only — no item text), plus the variant's
content-free log-probs for contextual calibration. Variants are chosen on the
``cal`` split; ``test`` is scored once, for the variant that ships.

Like the bake-off, it refuses to start unless klams' TEI routers are on the
GPU, and records whether they survived.
"""

import json
import random
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from bakeoff import VramPeak, gpu_used_mib, tei_pids
from build_evalset import route_descriptions
from evaldata import load

from kodds import normalize
from kodds.tasks import Task, load_tasks

OUT = HERE / "results" / "002"
N_CTX = 8192

ROUTE_HEAD = "You route work items to the homelab project that owns them.\n\n"
ROUTE_TAIL = "\n\nAnswer with the project name only."
ITEM = "Work item: {title}\n\n{content}"


def route_variants(task: Task) -> dict:
    """name -> Task. All keep the project list before the item, so it is cached."""
    listing = "Projects:\n{choice_list}\n\n"

    contracts = dict(task.descriptions)

    def with_examples(per_project: int, notes_cap: int = 0) -> Task:
        described = route_descriptions(contracts, per_project, notes_cap)
        return replace(task, descriptions=described)

    def with_notes(cap: int) -> Task:
        return replace(task, descriptions=route_descriptions(contracts, 0, cap))

    def fewshot(n: int) -> Task:
        train = load("route", "train")
        rng = random.Random(2026)
        rng.shuffle(train)
        shots = "".join(
            f"Work item: {r['inputs']['title']}\nProject: {r['label']}\n\n"
            for r in train[:n]
        )
        return replace(
            with_content,
            template=ROUTE_HEAD
            + listing
            + "Examples:\n\n"
            + shots.replace("{", "{{").replace("}", "}}")
            + ITEM
            + ROUTE_TAIL,
        )

    limits = {"title": 300, "content": 1500}
    title_only = replace(
        task,
        template=ROUTE_HEAD + listing + "Work item: {title}" + ROUTE_TAIL,
        limits={"title": 300},
    )
    with_content = replace(
        task, template=ROUTE_HEAD + listing + ITEM + ROUTE_TAIL, limits=limits
    )
    return {
        "title": lambda: title_only,
        "content": lambda: with_content,
        "content-short": lambda: replace(
            with_content, limits={"title": 300, "content": 500}
        ),
        "content-reversed": lambda: replace(
            with_content, choices=tuple(reversed(task.choices))
        ),
        "content-examples": lambda: replace(
            with_examples(3), template=with_content.template, limits=limits
        ),
        "content-examples5": lambda: replace(
            with_examples(5), template=with_content.template, limits=limits
        ),
        "content-examples-notes": lambda: replace(
            with_examples(3, notes_cap=200),
            template=with_content.template,
            limits=limits,
        ),
        "content-notes": lambda: replace(
            with_notes(400), template=with_content.template, limits=limits
        ),
        "content-fewshot": lambda: fewshot(16),
    }


def variants_for(task: Task) -> dict:
    if task.name == "route":
        return route_variants(task)
    return {"default": lambda: task}


def run(model_path: str, task_name: str, split: str, names: list[str]) -> None:
    from kodds.llama import load as load_model

    tei_before = tei_pids()
    if len(tei_before) < 2:
        sys.exit(f"expected klams' two TEI routers on the GPU, found {tei_before}")
    # Variants are built from the public definition, never the private overlay.
    task = load_tasks(overlay=False)[task_name]
    variants = variants_for(task)
    rows_in = load(task_name, split)
    peak = VramPeak()
    peak.start()
    baseline = gpu_used_mib()
    scorer = load_model(model_path, n_ctx=N_CTX)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        variant = variants[name]()
        cf = scorer.logprobs(variant.content_free(), variant.choices, variant.system)
        rows, tokens = [], []
        for row in rows_in:
            wanted = variant.inputs
            prompt = variant.render(
                **{k: v for k, v in row["inputs"].items() if k in wanted}
            )
            t = time.perf_counter()
            lp = scorer.logprobs(prompt, variant.choices, variant.system)
            latency = time.perf_counter() - t
            tokens.append(len(scorer.backend.tokenize(scorer.template(prompt))))
            rows.append(
                {"id": row["id"], "label": row["label"], "latency_s": latency}
                | {"logprobs": lp}
            )
        stem = f"{task_name}.{name}.{split}"
        (OUT / f"{stem}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (OUT / f"{task_name}.{name}.content_free.json").write_text(
            json.dumps(cf, indent=1) + "\n"
        )
        acc = statistics.fmean(
            max(r["logprobs"], key=r["logprobs"].__getitem__) == r["label"]
            for r in rows
        )
        brier = statistics.fmean(
            sum(
                (p - (c == r["label"])) ** 2
                for c, p in normalize(r["logprobs"]).items()
            )
            for r in rows
        )
        lat = sorted(r["latency_s"] for r in rows)
        print(
            f"{stem}: n={len(rows)} acc={acc:.3f} brier={brier:.3f} "
            f"p50={lat[len(lat) // 2] * 1000:.0f}ms p95={lat[int(len(lat) * 0.95)] * 1000:.0f}ms "
            f"prompt tokens max={max(tokens)} mean={statistics.fmean(tokens):.0f}",
            flush=True,
        )
    peak.stop()
    print(
        f"VRAM baseline {baseline} MiB, peak total {peak.total} MiB, "
        f"process {peak.mine} MiB; TEI untouched: {tei_pids() == tei_before}"
    )


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
