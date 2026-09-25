"""Measure a running kodds service's latency per task, as a caller sees it.

    uv run python evals/latency.py [URL]   # default http://127.0.0.1:7780

Uses ``latency_ms`` (inference inside the lock) from each response, over
distinct cal items so no call repeats a prompt:

- **warm**: the same task back to back. The task's fixed prefix (for route
  the ~7k-token project list) stays in llama.cpp's KV cache, so each call
  pays for its own input and the choices.
- **route after severity**: the prefix cache keeps only the previous
  prompt's prefix, so interleaving tasks makes route re-prefill its list.

The first call of the run is reported separately; it is "cold" only if the
service has served nothing else since it started.
"""

import json
import statistics
import sys
import urllib.request

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from evaldata import load

N = 8


def classify(url: str, row: dict) -> float:
    body = json.dumps({"task": row["task"], "inputs": row["inputs"]}).encode()
    req = urllib.request.Request(
        f"{url}/v1/classify", body, {"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["latency_ms"]


def summary(ms: list[float]) -> str:
    return f"median {statistics.median(ms):7.0f} ms  (min {min(ms):.0f}, max {max(ms):.0f}, n={len(ms)})"


def main(url: str) -> None:
    rows = {t: load(t, "cal") for t in ("route", "severity", "triage")}
    route, sev = iter(rows["route"]), iter(rows["severity"])
    print(f"first call (route): {classify(url, next(route)):.0f} ms")
    for task in ("route", "severity", "triage"):
        it = (
            route
            if task == "route"
            else sev
            if task == "severity"
            else iter(rows[task])
        )
        classify(url, next(it))  # prime this task's prefix
        print(f"{task:>8} warm: {summary([classify(url, next(it)) for _ in range(N)])}")
    interleaved = []
    for _ in range(N):
        classify(url, next(sev))
        interleaved.append(classify(url, next(route)))
    print(f"route after severity: {summary(interleaved)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7780")
