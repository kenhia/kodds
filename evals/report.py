"""Print a markdown comparison of every evals/results/*.json.

The last table asks whether a model's miscalibration is *fixable*: one
temperature T per model (probabilities = softmax(logprobs / T)), fitted on
alternate items and scored on the others, both ways round. A model whose
Brier drops a lot under T is overconfident in a way a calibration step can
repair; one whose Brier barely moves is wrong, not merely loud.
"""

import json
import math
import statistics
from pathlib import Path

RESULTS = Path(__file__).with_name("results")
TEMPERATURES = [0.5 * 1.1**k for k in range(60)]  # 0.5 .. ~150


def tempered(logprobs: dict[str, float], t: float) -> dict[str, float]:
    peak = max(logprobs.values())
    w = {c: math.exp((v - peak) / t) for c, v in logprobs.items()}
    z = sum(w.values())
    return {c: x / z for c, x in w.items()}


def brier_at(rows: list[dict], t: float) -> float:
    return statistics.fmean(
        sum((p - (c == r["label"])) ** 2 for c, p in tempered(r["logprobs"], t).items())
        for r in rows
    )


def nll_at(rows: list[dict], t: float) -> float:
    return statistics.fmean(
        -math.log(max(tempered(r["logprobs"], t)[r["label"]], 1e-300)) for r in rows
    )


def fit_t(rows: list[dict]) -> float:
    return min(TEMPERATURES, key=lambda t: nll_at(rows, t))


def cross_validated(rows: list[dict]) -> tuple[float, float]:
    """(Brier with T fitted on the other half, T fitted on everything)."""
    a, b = rows[0::2], rows[1::2]
    scored = brier_at(b, fit_t(a)) * len(b) + brier_at(a, fit_t(b)) * len(a)
    return scored / len(rows), fit_t(rows)


def main() -> None:
    runs = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))]
    print(
        "| model | peak VRAM (GPU total / process) | load s | p50 / p95 ms "
        "| acc | Brier | ECE |"
    )
    print("|---|---|---|---|---|---|---|")
    for r in runs:
        v, lat, o = r["vram"], r["latency_s"], r["overall"]
        print(
            f"| {r['model']} | {v['peak_total_mib'] / 1024:.1f} / "
            f"{v['peak_process_mib'] / 1024:.1f} GiB | {r['load_s']:.1f} "
            f"| {lat['p50'] * 1000:.0f} / {lat['p95'] * 1000:.0f} "
            f"| {o['accuracy']:.2f} | {o['brier']:.3f} | {o['ece']:.3f} |"
        )
    print()
    tasks = sorted({t for r in runs for t in r["by_task"]})
    print("| model | " + " | ".join(f"{t} acc / Brier / ECE" for t in tasks) + " |")
    print("|---|" + "---|" * len(tasks))
    for r in runs:
        cells = [
            f"{r['by_task'][t]['accuracy']:.2f} / {r['by_task'][t]['brier']:.3f} / "
            f"{r['by_task'][t]['ece']:.3f}"
            for t in tasks
        ]
        print(f"| {r['model']} | " + " | ".join(cells) + " |")
    print()
    print("| model | Brier raw | Brier, one T (2-fold CV) | T (all items) |")
    print("|---|---|---|---|")
    for r in runs:
        if "logprobs" not in r["rows"][0]:
            continue
        cv, t = cross_validated(r["rows"])
        print(f"| {r['model']} | {r['overall']['brier']:.3f} | {cv:.3f} | {t:.1f} |")


if __name__ == "__main__":
    main()
