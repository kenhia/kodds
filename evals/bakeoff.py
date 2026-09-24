"""Run evals/evalset.jsonl against one GGUF and write its results.

    uv run python evals/bakeoff.py ~/models/gguf/Qwen3-4B-Q8_0.gguf

One model per process, so its VRAM is released before the next one loads.
Measures load time, per-call latency, peak VRAM (whole GPU and this process,
polled from nvidia-smi), accuracy, Brier score and ECE — overall and per task
— and refuses to start if klams' text-embeddings-router processes are not
running, since the point is to measure alongside them.
"""

import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
EVALSET = HERE / "evalset.jsonl"
RESULTS = HERE / "results"
BINS = 10


def gpu_processes() -> dict[int, tuple[str, int]]:
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    procs = {}
    for line in out.strip().splitlines():
        pid, name, mib = (x.strip() for x in line.split(","))
        procs[int(pid)] = (name, int(mib))
    return procs


def gpu_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return int(out.strip().splitlines()[0])


def tei_pids() -> set[int]:
    return {
        pid
        for pid, (name, _) in gpu_processes().items()
        if "text-embeddings-router" in name
    }


class VramPeak(threading.Thread):
    def __init__(self, interval: float = 0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.total = 0
        self.mine = 0
        self._stop = threading.Event()

    def run(self):
        me = os.getpid()
        while not self._stop.is_set():
            self.total = max(self.total, gpu_used_mib())
            self.mine = max(self.mine, gpu_processes().get(me, ("", 0))[1])
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()
        self.join()


def brier(probs: dict[str, float], label: str) -> float:
    return sum((p - (c == label)) ** 2 for c, p in probs.items())


def ece(confidences: list[float], correct: list[bool]) -> float:
    """Top-label expected calibration error, equal-width bins."""
    n = len(confidences)
    total = 0.0
    for b in range(BINS):
        lo, hi = b / BINS, (b + 1) / BINS
        idx = [
            i
            for i, c in enumerate(confidences)
            if lo < c <= hi or (b == 0 and c == 0.0)
        ]
        if idx:
            conf = statistics.fmean(confidences[i] for i in idx)
            acc = statistics.fmean(correct[i] for i in idx)
            total += len(idx) / n * abs(conf - acc)
    return total


def summarise(rows: list[dict]) -> dict:
    conf = [r["confidence"] for r in rows]
    correct = [r["correct"] for r in rows]
    return {
        "n": len(rows),
        "accuracy": statistics.fmean(correct),
        "brier": statistics.fmean(r["brier"] for r in rows),
        "ece": ece(conf, correct),
        "mean_confidence": statistics.fmean(conf),
    }


def main(model_path: str) -> None:
    from kodds import normalize
    from kodds.llama import load

    items = [json.loads(line) for line in EVALSET.read_text().splitlines()]
    tei_before = tei_pids()
    if len(tei_before) < 2:
        sys.exit(f"expected klams' two TEI routers on the GPU, found {tei_before}")
    baseline = gpu_used_mib()

    peak = VramPeak()
    peak.start()
    t0 = time.perf_counter()
    scorer = load(model_path)
    load_s = time.perf_counter() - t0

    scorer.score("warm-up", ["a", "b"])
    rows: list[dict[str, Any]] = []
    for item in items:
        t = time.perf_counter()
        logprobs = scorer.logprobs(item["prompt"], item["choices"])
        latency = time.perf_counter() - t
        probs = normalize(logprobs)
        pick = max(probs, key=probs.__getitem__)
        rows.append(
            {
                "task": item["task"],
                "label": item["label"],
                "pick": pick,
                "correct": pick == item["label"],
                "confidence": probs[pick],
                "p_label": probs[item["label"]],
                "brier": brier(probs, item["label"]),
                "latency_s": latency,
                "logprobs": logprobs,
            }
        )
    peak.stop()
    tei_after = tei_pids()

    latencies: list[float] = sorted(r["latency_s"] for r in rows)
    result = {
        "model": Path(model_path).name,
        "load_s": load_s,
        "vram": {
            "baseline_mib": baseline,
            "peak_total_mib": peak.total,
            "peak_process_mib": peak.mine,
            "tei_untouched": tei_before == tei_after,
        },
        "latency_s": {
            "mean": statistics.fmean(latencies),
            "p50": latencies[len(latencies) // 2],
            "p95": latencies[int(len(latencies) * 0.95)],
        },
        "overall": summarise(rows),
        "by_task": {
            task: summarise([r for r in rows if r["task"] == task])
            for task in sorted({r["task"] for r in rows})
        },
        "rows": rows,
    }
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{Path(model_path).stem}.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    brief = {k: v for k, v in result.items() if k != "rows"}
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
