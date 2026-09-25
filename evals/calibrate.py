"""Compare calibration methods per task: fit on ``cal``, report on ``test``.

    uv run python evals/calibrate.py route=content severity=default triage=default
    uv run python evals/calibrate.py --write Qwen3-14B-Q4_K_M.gguf route=content ...

Reads ``evals/results/002/<task>.<variant>.{cal,test}.jsonl`` and the
variant's content-free log-probs, all written by ``sweep.py`` — so this needs
no GPU. Every method is ``softmax((logprob + bias) / T)``:

- ``raw``      — T = 1, no bias.
- ``global-T`` — one T for every task (001's baseline), fitted on all cal rows.
- ``task-T``   — one T per task.
- ``cc``       — contextual calibration: bias = −(content-free log-prob), T = 1.
- ``cc+T``     — the same bias, with a T fitted on top.
- ``bias+T``   — a per-choice bias vector and T fitted together, L2-shrunk
  towards zero (the bias has one free parameter per choice, and small tasks
  overfit it).
- ``bias+T/light`` — the same with a hundredth of the shrinkage (measured
  on route: level with no shrinkage in CV, with a tamer bias range). Which of
  the two wins is itself left to cross-validation. In both, a choice with no
  cal items keeps the median bias rather than being fitted towards −∞.

Every fitted T is held to T >= 1. With a few dozen cal items a task can be
perfectly separable (triage is), and then the likelihood drives T towards 0,
sharpening an already-certain model so that one miss costs several nats.

The method written for a task is ``global-T`` unless another beats it by more
than one standard error of the paired per-item NLL under 5-fold
cross-validation *within* cal, and a task with no cal errors always keeps it.
Test is only ever reported, never chosen on.
ECE on a few dozen items is noisy: lean on Brier and NLL.
"""

import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RESULTS = HERE / "results" / "002"
from kodds.tasks import Calibration, load_tasks, save_calibration

BINS = 10
FOLDS = 5
L2 = 0.01  # per-choice bias shrinkage for ``bias+T``, against the mean NLL


@dataclass
class Split:
    choices: list[str]
    lp: np.ndarray  # (n, k) summed log-probs
    y: np.ndarray  # (n,) label index
    ids: list[str]

    @classmethod
    def read(cls, path: Path, choices: list[str]) -> "Split":
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        lp = np.array([[r["logprobs"][c] for c in choices] for r in rows])
        y = np.array([choices.index(r["label"]) for r in rows])
        return cls(choices, lp, y, [r["id"] for r in rows])

    def take(self, idx: np.ndarray) -> "Split":
        return Split(
            self.choices, self.lp[idx], self.y[idx], [self.ids[i] for i in idx]
        )


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def probs(s: Split, bias: np.ndarray | float, t: float) -> np.ndarray:
    return softmax((s.lp + bias) / t)


def fit(
    split: Split,
    bias0: np.ndarray,
    *,
    fit_t: bool,
    fit_bias: bool,
    l2: float = L2,
) -> tuple[np.ndarray, float]:
    """Minimise mean NLL (+ L2 on the fitted part of the bias) by Adam."""
    lp, y = split.lp, split.y
    n, k = lp.shape
    delta, log_t = np.zeros(k), 0.0
    m, v = np.zeros(k + 1), np.zeros(k + 1)
    lr, b1, b2 = 0.05, 0.9, 0.999
    onehot = np.eye(k)[y]
    # Only choices with cal items get a bias of their own: a choice cal never
    # shows would otherwise be pushed towards -inf (every gradient step says
    # "less"), making it unroutable. It is left at the median bias instead.
    seen = np.bincount(y, minlength=k) > 0
    for step in range(1, 3001):
        t = np.exp(log_t)
        z = (lp + bias0 + delta) / t
        g = (softmax(z) - onehot) / n  # dNLL/dz
        grad_delta = (
            (g.sum(axis=0) / t + 2 * l2 * delta) * seen if fit_bias else np.zeros(k)
        )
        grad_log_t = -(g * z).sum() if fit_t else 0.0
        grad = np.append(grad_delta, grad_log_t)
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad**2
        upd = lr * (m / (1 - b1**step)) / (np.sqrt(v / (1 - b2**step)) + 1e-9)
        delta -= upd[:k]
        log_t = float(np.clip(log_t - upd[k], 0.0, 5.0))  # T >= 1: never sharpen
    if fit_bias:
        delta = np.where(seen, delta, np.median(delta[seen]))
    return bias0 + delta, float(np.exp(log_t))


def fit_shared_t(splits: list[Split]) -> float:
    """One T for several tasks (their choice counts differ): min pooled NLL."""
    grid = np.exp(np.linspace(0.0, 5.0, 501))
    total = [sum(nll(probs(s, 0.0, t), s.y) * len(s.y) for s in splits) for t in grid]
    return float(grid[int(np.argmin(total))])


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    n = len(y)
    pick = p.argmax(axis=1)
    conf = p.max(axis=1)
    correct = pick == y
    onehot = np.eye(p.shape[1])[y]
    table = []
    ece = 0.0
    for b in range(BINS):
        lo, hi = b / BINS, (b + 1) / BINS
        idx = (conf > lo) & (conf <= hi)
        if idx.any():
            gap = abs(conf[idx].mean() - correct[idx].mean())
            ece += idx.sum() / n * gap
            table.append(
                (
                    f"{lo:.1f}-{hi:.1f}",
                    int(idx.sum()),
                    conf[idx].mean(),
                    correct[idx].mean(),
                )
            )
    return {
        "n": n,
        "accuracy": float(correct.mean()),
        "brier": float(((p - onehot) ** 2).sum(axis=1).mean()),
        "nll": float(-np.log(np.clip(p[np.arange(n), y], 1e-300, None)).mean()),
        "ece": float(ece),
        "mean_confidence": float(conf.mean()),
        "reliability": table,
    }


def nll(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-300, None)).mean())


DEFAULT = "global-T"
METHODS = ["raw", "global-T", "task-T", "cc", "cc+T", "bias+T", "bias+T/light"]


def fit_method(method: str, cal: Split, cf: np.ndarray, pooled: list[Split]):
    zero = np.zeros(len(cal.choices))
    match method:
        case "raw":
            return zero, 1.0
        case "global-T":
            return zero, fit_shared_t(pooled)
        case "task-T":
            return fit(cal, zero, fit_t=True, fit_bias=False)
        case "cc":
            return -cf, 1.0
        case "cc+T":
            return fit(cal, -cf, fit_t=True, fit_bias=False)
        case "bias+T":
            return fit(cal, zero, fit_t=True, fit_bias=True)
        case "bias+T/light":
            return fit(cal, zero, fit_t=True, fit_bias=True, l2=L2 / 100)
    raise ValueError(method)


def cross_validated(
    method: str, cal: Split, cf: np.ndarray, others: list[Split]
) -> tuple[np.ndarray, float]:
    """(per-item NLL, accuracy) on cal, each fold scored by params fitted on the rest."""
    order = np.random.default_rng(2026).permutation(len(cal.y))
    folds = np.array_split(order, FOLDS)
    losses = np.zeros(len(cal.y))
    correct = 0
    for i, held in enumerate(folds):
        train = cal.take(np.concatenate([f for j, f in enumerate(folds) if j != i]))
        bias, t = fit_method(method, train, cf, [train, *others])
        p = probs(cal.take(held), bias, t)
        losses[held] = -np.log(
            np.clip(p[np.arange(len(held)), cal.y[held]], 1e-300, None)
        )
        correct += int((p.argmax(axis=1) == cal.y[held]).sum())
    return losses, correct / len(cal.y)


def choose(cv_losses: dict[str, np.ndarray], cal_errors: int) -> str:
    """``DEFAULT`` unless a method beats it by more than one standard error.

    With a few dozen cal items, the lowest CV NLL among seven methods is
    mostly noise; a task earns its own method only on clear evidence. A task
    with no errors on cal has none to offer: every T > 1 then costs a little
    on every item, so "sharper is better" wins consistently and means
    nothing. It keeps the default, fitted with the other tasks' errors.
    """
    if cal_errors == 0:
        return DEFAULT
    base = cv_losses[DEFAULT]
    best, margin = DEFAULT, 0.0
    for method, losses in cv_losses.items():
        diff = base - losses
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        if diff.mean() - se > margin:
            best, margin = method, diff.mean() - se
    return best


def swept_prompt_hash(task: str, variant: str) -> str:
    """The prompt hash of a variant as ``sweep.py`` built it."""
    sys.path.insert(0, str(HERE))
    from sweep import variants_for

    public = load_tasks(overlay=False)[task]
    return variants_for(public)[variant]().prompt_hash()


def main(args: list[str]) -> None:
    write_model = None
    if args and args[0] == "--write":
        write_model, args = args[1], args[2:]
    tasks = load_tasks()
    chosen = dict(a.split("=", 1) for a in args)
    data = {}
    for name, variant in chosen.items():
        choices = list(tasks[name].choices)
        cf = json.loads((RESULTS / f"{name}.{variant}.content_free.json").read_text())
        data[name] = (
            Split.read(RESULTS / f"{name}.{variant}.cal.jsonl", choices),
            Split.read(RESULTS / f"{name}.{variant}.test.jsonl", choices),
            np.array([cf[c] for c in choices]),
        )
    report: dict = {}
    for name, (cal, test, cf) in data.items():
        others = [d[0] for n, d in data.items() if n != name]
        report[name] = {"variant": chosen[name], "methods": {}}
        print(
            f"\n### {name} ({chosen[name]}) — cal n={len(cal.y)}, test n={len(test.y)}\n"
        )
        print(
            "| method | T | CV NLL (cal) | test acc | Brier | NLL | ECE | mean conf |"
        )
        print("|---|---|---|---|---|---|---|---|")
        cv_losses: dict[str, np.ndarray] = {}
        for method in METHODS:
            bias, t = fit_method(method, cal, cf, [cal, *others])
            m = metrics(probs(test, bias, t), test.y)
            losses, cv_acc = cross_validated(method, cal, cf, others)
            cv_losses[method] = losses
            cv = float(losses.mean())
            report[name]["methods"][method] = {
                "t": t,
                "bias": bias.tolist(),
                "cv_nll": cv,
                "cv_accuracy": cv_acc,
                "test": m,
            }
            print(
                f"| {method} | {t:.2f} | {cv:.3f} / {cv_acc:.3f} "
                f"| {m['accuracy']:.3f} | {m['brier']:.3f} | {m['nll']:.3f} "
                f"| {m['ece']:.3f} | {m['mean_confidence']:.2f} |"
            )
        cal_errors = int((cal.lp.argmax(axis=1) != cal.y).sum())
        best = choose(cv_losses, cal_errors)
        report[name]["chosen"] = best
        print(
            f"\nchosen on cal (global-T unless beaten by >1 SE of CV NLL): **{best}**; test reliability:\n"
        )
        print("| confidence | n | mean conf | accuracy |")
        print("|---|---|---|---|")
        for label, n, conf, acc in report[name]["methods"][best]["test"]["reliability"]:
            print(f"| {label} | {n} | {conf:.2f} | {acc:.2f} |")
        if write_model:
            shipped = tasks[name]
            if swept_prompt_hash(name, chosen[name]) != shipped.prompt_hash():
                sys.exit(
                    f"{name}: variant {chosen[name]!r} is not the prompt tasks/ "
                    "would serve — update the task (or its overlay) first"
                )
            r = report[name]["methods"][best]
            cal_json = Calibration(
                temperature=r["t"],
                bias={c: b for c, b in zip(cal.choices, r["bias"], strict=True) if b},
                choices=tuple(cal.choices),
                method=best,
                prompt_sha256=shipped.prompt_hash(),
                provenance={
                    "fitted": str(datetime.now(UTC).date()),
                    "prompt_variant": chosen[name],
                    "fit_split": "cal",
                    "fit_n": len(cal.y),
                    "cv_nll_cal": round(r["cv_nll"], 4),
                    "test": {
                        k: round(v, 4)
                        for k, v in r["test"].items()
                        if k in ("n", "accuracy", "brier", "nll", "ece")
                    },
                    "test_raw": {
                        k: round(v, 4)
                        for k, v in report[name]["methods"]["raw"]["test"].items()
                        if k in ("accuracy", "brier", "nll", "ece")
                    },
                },
            )
            print(f"wrote {save_calibration(tasks[name], write_model, cal_json)}")
    (RESULTS / "calibration.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
