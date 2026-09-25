"""Build kodds' eval data: fixed calibration / test (/ train) splits per task.

Every row is ``{task, id, split, inputs, label}``; ``inputs`` are the task's
template fields (``tasks/<task>.toml``), so a row can be rendered under any
prompt variant being compared.

- ``route``    — which korg project owns a work item. Real items from korg
  history (title + content), labelled by the project each is filed under.
  kodds is a public repo and korg is not, so only the *assignment* is
  committed — ``evals/splits/route.json``, wi_number → label and split — and
  the text is fetched from korg into git-ignored ``.scratch/evalset/`` when
  this runs. (See ``evals/README.md``.)
- ``severity`` — kmon's three-way call on a finding, labelled by hand against
  the rubric in kmon/controller.py. Still hand-made: kmon doesn't log its calls.
- ``triage``   — legitimate / spam / phishing, synthetic.

Splits are fixed once, seeded, and never refitted: ``cal`` fits calibration
(and chooses prompts), ``test`` only reports numbers, and route's ``train`` is
the reserve for fine-tuning (sprints/planning/fine-tuning.md) and the source
of few-shot examples, so neither ever borrows from test.

    uv run python evals/build_evalset.py projects  # refresh route's choices from korg
    uv run python evals/build_evalset.py splits    # ONE-TIME: fix route's split
    uv run python evals/build_evalset.py           # write the JSONL
    uv run python evals/build_evalset.py overlay   # route's private descriptions
"""

import json
import os
import random
import re
import sys
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from evaldata import PRIVATE, PUBLIC, SPLITS, load

KORG = os.environ.get("KORG_URL", "http://kubsdb:5674")
ROUTE_TOML = ROOT / "tasks" / "route.toml"
SEED = 2026
PER_SPLIT_CAP = 12  # route items per project in each of cal and test
# Active korg projects that are never a destination for real work.
NOT_ROUTABLE = {"eval"}

# --- severity / triage ------------------------------------------------------

SEVERITY_ITEMS = {
    "ok": [
        "kai: all 142 systemd units active; no failed units; load average 0.8 on 32 cores.",
        "kubs0: root filesystem at 41% used, up from 40% yesterday.",
        "kubsdb: postgres, mongo and redis containers all healthy; nightly backup completed at 03:12 in 4m.",
        "rpi53: kpidash.service restarted once at 04:00 by its scheduled timer, as configured.",
        "kai: AMD-Vi Completion-Wait timeout logged once at boot — known benign firmware quirk, already investigated.",
        "kubs0: GPU at 18% memory (2.9 GB), two text-embeddings-router processes, both expected.",
        "ksandbox: 6 trial containers created and destroyed overnight — host is substrate collect_mode, churn is by design.",
        "kai: apt reports 3 upgradable packages, none security.",
        "kubsdb: Grafana image one minor version behind upstream; no advisories.",
        "kai: tailscale up, all 9 peers reachable, latency under 3 ms.",
        "kubs0: klams-scanner.timer last ran 02:00, indexed 214 changed files, exit 0.",
        "rpi53: CPU temperature 52°C, steady across the day.",
        "kai: kmon.timer fired at 05:30 and the report was filed.",
        "kubsdb: disk SMART attributes unchanged; 0 reallocated sectors.",
        "kai: journal size 1.9 GB, under the 4 GB cap.",
    ],
    "attention": [
        "kubs0: /sys/firmware/efi/efivars at 82% (pressured); unchanged for 12 days.",
        "kai: run-u558.service failed (failed/failed) — a transient unit, origin unknown from the bundle.",
        "kubsdb: root filesystem at 78%, up 6 points in a week; at this rate full in about four weeks.",
        "kai: a user unit kdeskdash-claude-poll.timer is running but absent from the k-homelab manifest.",
        "rpi53: NetworkManager-wait-online.service failed at boot; network came up anyway.",
        "kai: openipmi.service failing (failed/failed) since the last kernel update.",
        "kubs0: a listener on :8811 has no matching manifest entry and no owning project in korg.",
        "kai: nvidia-cdi-refresh.service failed after the driver update; containers still start.",
        "kubsdb: nightly-backup.service took 47 minutes, up from a usual 4; completed successfully.",
        "kai: 212 apt packages upgradable including 14 security updates, pending for 9 days.",
        "kubs0: kaed binary on disk is 0.4.1 but the package store's current release is 0.5.0.",
        "rpi53: SD card write count up 30% this week; card is 3 years old.",
        "kai: a Docker image 58 GB in size appeared under /var/lib/docker on the root volume, not the AI volume.",
        "kubsdb: redis reports 2 rejected connections with a stale password from rpi53.",
        "kai: the kcard-backup.service unit failed twice this week, succeeded on retry both times.",
    ],
    "problem": [
        "kubsdb: postgres container exited 137 (OOM) at 02:14 and has not restarted; korg and klams are down.",
        "kai: root filesystem at 99% used, 1.2 GB free, growing 3 GB/hour.",
        "kubs0: klams service unreachable for 3 hours; health endpoint times out.",
        "kai: global OOM killer fired 4 times in the last hour; sshd was killed once.",
        "kubsdb: SMART reports 1,204 reallocated sectors, up from 0 yesterday; pending sectors 88.",
        "rpi53: unreachable on tailscale and LAN for 6 hours; last seen with CPU at 85°C.",
        "kubs0: nightly backups have failed for 5 consecutive days; last good backup is 6 days old.",
        "kai: sshd config changed overnight to PermitRootLogin yes and PasswordAuthentication yes; no matching work item.",
        "kubsdb: the ZFS pool is DEGRADED — one of two mirror disks FAULTED.",
        "kai: a process listening on 0.0.0.0:4444 owned by an unknown user 'svc_tmp' created today.",
        "kubs0: GPU fallen off the bus (Xid 79); all CUDA workloads failing.",
        "kubsdb: redis is accepting unauthenticated connections from the LAN after a config reset.",
        "kai: the kvllm server is crash-looping every 20 s for 2 hours, and kmon, kyac and klams-mind all depend on it.",
        "kubsdb: TLS certificate for the korg endpoint expired 3 hours ago; every client is refusing it.",
        "kubs0: /home volume remounted read-only after ext4 errors.",
    ],
}


TRIAGE_ITEMS = {
    "legitimate": [
        "From: GitHub <noreply@github.com> — [kenhia/korg] PR #412 merged: 'list_projects gains omitted counts'.",
        "From: Tailscale <billing@tailscale.com> — Your receipt for September. Personal plan, $0.00.",
        "From: Mom — Are you still coming Sunday? Dad wants to show you the new brickshooter levels.",
        "From: Hugging Face <no-reply@huggingface.co> — Qwen/Qwen3-8B-GGUF has a new revision you follow.",
        "From: Amazon <shipment-tracking@amazon.com> — Your order of 2 x Crucial 32GB DDR5 has shipped; arriving Thursday.",
        "From: Ubuntu Security <security@ubuntu.com> — USN-7712-1: OpenSSL vulnerabilities, update packages.",
        "From: Dr. Patel's office — Reminder: your appointment is Tuesday at 2:30 PM. Reply C to confirm.",
        "From: Cloudflare <noreply@notify.cloudflare.com> — Your domain renewal for example-home.net completed.",
        "From: Raspberry Pi Ltd <newsletter@raspberrypi.com> — New in the shop: the Pi 5 16GB. Unsubscribe any time.",
        "From: Jen (neighbour) — Your package ended up on our porch, I'll drop it over tonight.",
        "From: Backblaze <no-reply@backblaze.com> — Weekly report: 2.1 TB backed up, no errors.",
        "From: PyPI <noreply@pypi.org> — New login to your account from Seattle, WA using a security key.",
        "From: Costco Photo — Your prints are ready for pickup at store #0411.",
        "From: GitHub <noreply@github.com> — Dependabot opened 3 pull requests on kenhia/kvllm.",
        "From: City Water Utility — Your September statement is available: $48.20, autopay on the 15th.",
    ],
    "spam": [
        "From: Deals Daily <promo@dealz-blast.biz> — 🔥 90% OFF designer watches TODAY ONLY!!! Click to shop.",
        "From: Crypto Signals VIP — Our members made 4,000% last month. Join the Telegram group now.",
        "From: Dr. Slim <info@fastketo-pills.info> — Lose 30 lbs in 30 days with no diet. Order today.",
        "From: SEO Experts — We noticed your site isn't ranking on page 1. Our team can fix that for $99.",
        "From: Casino Royale Online — Your 200 free spins are waiting. Bet now!",
        "From: Hot Singles Near You — 3 women in your area want to meet tonight.",
        "From: Global Traders <offer@b2b-leads.cn> — Wholesale LED strips, MOQ 10,000, factory price.",
        "From: Lucky Winner Notice — Congratulations, you have been selected for a survey reward. Claim your gift card!",
        "From: Solar Savings Team — Homeowners in your ZIP are getting FREE solar panels. See if you qualify.",
        "From: Replica Watches — Rolex, Omega, Tag Heuer replicas, 1:1 quality, worldwide shipping.",
        "From: Web Design Agency — Hi, I can redesign your website at an affordable price. Interested?",
        "From: Extended Warranty Dept — Final notice: your vehicle's factory warranty is about to expire.",
        "From: Miracle Hair — Regrow a full head of hair in 2 weeks — doctors hate this trick.",
        "From: Mega Lottery International — Buy 5 tickets, get 5 free for this week's $400M draw.",
        "From: List Broker — Purchase 2 million verified B2B email contacts for $149.",
    ],
    "phishing": [
        "From: GitHub Security <security@github-verify-account.com> — Unusual sign-in detected. Verify your account within 24 hours or it will be suspended: http://github-verify-account.com/login",
        "From: PayPal <service@paypa1-support.com> — Your account has been limited. Log in here to restore access and confirm your card details.",
        "From: Microsoft 365 Admin — Your mailbox is 99% full. Sign in at the link below to keep receiving mail.",
        "From: Tailscale <admin@tailscale-security.net> — A new device joined your tailnet. If this wasn't you, re-enter your credentials here.",
        "From: IT Helpdesk — Your password expires today. Reply with your current password to keep it active.",
        "From: Amazon <account-update@amazon-billing-help.com> — Your order was cancelled due to a payment failure. Update your billing information.",
        "From: Chase Bank — We detected a suspicious charge of $1,249.00. Call this number and give the code we text you.",
        "From: DocuSign <dse@docusign-review.co> — Ken, you have a document to review and sign: Invoice_4471.html",
        "From: Hugging Face <no-reply@hf-tokens.support> — Your access token was leaked. Paste it here so we can revoke it.",
        "From: USPS — Your package could not be delivered. Pay a $1.99 redelivery fee: usps-redelivery-track.info",
        "From: Apple ID — Your Apple ID was used to sign in on a new iPhone in Moscow. Unlock your account at apple-id-unlock.com",
        "From: CEO (via Gmail) — I'm in a meeting, need you to buy 5 $200 gift cards and send me the codes asap.",
        "From: Cloudflare <billing@cloudf1are.com> — Your domain will be deleted tomorrow. Renew now with the attached payment form.",
        "From: Google Workspace — Shared file 'Payroll_Q3.xlsx' — sign in with your Google password to view.",
        "From: Coinbase Support — Your wallet is frozen. Enter your recovery phrase to verify ownership.",
    ],
}


def stratified(items: dict[str, list[str]], n_cal: int) -> list[tuple[str, str, str]]:
    """(label, text, split) per item: ``n_cal`` of each label to cal, rest test."""
    out = []
    for label, texts in items.items():
        order = list(range(len(texts)))
        random.Random(f"{SEED}:{label}").shuffle(order)
        for rank, i in enumerate(order):
            out.append((label, texts[i], "cal" if rank < n_cal else "test"))
    return out


# --- route (korg) -----------------------------------------------------------


def korg_get(path: str):
    with urllib.request.urlopen(f"{KORG}{path}", timeout=60) as r:
        return json.load(r)


def korg_work_items() -> list[dict]:
    items, offset = [], 0
    while True:
        page = korg_get(
            f"/api/work-items?limit=500&offset={offset}&wi_status=all&archived=all"
        )
        items += page["items"]
        offset += 500
        if offset >= page["total"]:
            return items


def route_choices() -> list[str]:
    import tomllib

    return list(tomllib.loads(ROUTE_TOML.read_text())["choices"])


def refresh_projects() -> None:
    """Rewrite route.toml's trailing ``[choices]`` table from korg's active projects."""
    projects = korg_get("/api/projects")
    active = [
        p
        for p in projects
        if p.get("status", "active") == "active" and p["name"] not in NOT_ROUTABLE
    ]
    table = "".join(
        f"{json.dumps(p['name'])} = {json.dumps(p['description'], ensure_ascii=False)}\n"
        for p in sorted(active, key=lambda p: p["name"].lower())
    )
    text = ROUTE_TOML.read_text()
    head = text[: text.index("[choices]\n")]
    ROUTE_TOML.write_text(f"{head}[choices]\n{table}")
    print(f"{len(active)} routable projects written to {ROUTE_TOML}")


def fix_route_splits() -> None:
    path = SPLITS / "route.json"
    if path.exists():
        sys.exit(f"{path} exists — splits are fixed once; delete it deliberately")
    choices = set(route_choices())
    by_project: dict[str, list[int]] = {}
    for wi in korg_work_items():
        if (
            wi["project"] in choices
            and not wi["archived"]
            and (wi["content"] or "").strip()
        ):
            by_project.setdefault(wi["project"], []).append(wi["wi_number"])
    rows = []
    for project, wis in sorted(by_project.items()):
        wis.sort()
        random.Random(f"{SEED}:{project}").shuffle(wis)
        k = min(PER_SPLIT_CAP, len(wis) * 2 // 5)
        for rank, wi in enumerate(wis):
            split = "cal" if rank < k else "test" if rank < 2 * k else "train"
            rows.append((wi, project, split))
    rows.sort()
    SPLITS.mkdir(exist_ok=True)
    body = ",\n".join(json.dumps(r) for r in rows)
    path.write_text(
        "{\n"
        f' "seed": {SEED},\n'
        f' "snapshot": "{datetime.now(UTC).date()}",\n'
        f' "rule": "per project: min({PER_SPLIT_CAP}, 40%) to cal, as many to test, '
        'the rest to train",\n'
        f' "items": [\n{body}\n ]\n}}\n'
    )
    print(f"{len(rows)} route items assigned: {Counter(r[2] for r in rows)}")


def route_rows() -> list[dict]:
    fixed = json.loads((SPLITS / "route.json").read_text())["items"]
    korg = {wi["wi_number"]: wi for wi in korg_work_items()}
    rows, missing, refiled = [], [], []
    for wi, label, split in fixed:
        item = korg.get(wi)
        if item is None:
            missing.append(wi)
            continue
        if item["project"] != label:
            refiled.append(wi)  # keep the fixed label; say so
        rows.append(
            {
                "task": "route",
                "id": f"wi-{wi}",
                "split": split,
                "inputs": {"title": item["title"], "content": clean(item["content"])},
                "label": label,
            }
        )
    if missing or refiled:
        print(
            f"route: {len(missing)} gone from korg {missing}, {len(refiled)} refiled since the split {refiled}"
        )
    return rows


def clean(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text.strip())


def route_descriptions(
    contracts: dict[str, str], per_project: int = 3, notes_cap: int = 200
) -> dict[str, str]:
    """Each project's contract, a notes excerpt and example titles from train.

    What #3193 measured best. It is private korg content, so it ships as the
    git-ignored overlay ``tasks/private/route.json``, never in route.toml.
    """
    train = load("route", "train")
    random.Random(SEED).shuffle(train)
    titles: dict[str, list[str]] = {}
    for row in train:
        picked = titles.setdefault(row["label"], [])
        if len(picked) < per_project:
            picked.append(row["inputs"]["title"])
    notes = {p["name"]: p.get("notes") or "" for p in korg_get("/api/projects")}
    described = {}
    for name, contract in contracts.items():
        extra = " ".join(notes.get(name, "").split())
        if notes_cap and extra:
            cut = extra[:notes_cap] + ("…" if len(extra) > notes_cap else "")
            contract = f"{contract} {cut}"
        described[name] = contract + "".join(
            f'\n    e.g. "{t}"' for t in titles.get(name, [])
        )
    return described


def write_route_overlay() -> None:
    import tomllib

    contracts = tomllib.loads(ROUTE_TOML.read_text())["choices"]
    path = ROOT / "tasks" / "private" / "route.json"
    path.parent.mkdir(exist_ok=True)
    body = {"descriptions": route_descriptions(contracts)}
    path.write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path.relative_to(ROOT)} (git-ignored)")


# --- output -----------------------------------------------------------------


def write(rows: list[dict], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["task"], row["split"]), []).append(row)
    for (task, split), group in sorted(groups.items()):
        path = directory / f"{task}.{split}.jsonl"
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in group)
        )
        print(f"{path.relative_to(ROOT)}: {len(group)}")


def main() -> None:
    public = []
    for task, field, items in (
        ("severity", "finding", SEVERITY_ITEMS),
        ("triage", "message", TRIAGE_ITEMS),
    ):
        for n, (label, text, split) in enumerate(stratified(items, n_cal=7)):
            public.append(
                {
                    "task": task,
                    "id": f"{task}-{n}",
                    "split": split,
                    "inputs": {field: text},
                    "label": label,
                }
            )
    write(public, PUBLIC)
    write(route_rows(), PRIVATE)


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "build"
    {
        "projects": refresh_projects,
        "splits": fix_route_splits,
        "build": main,
        "overlay": write_route_overlay,
    }[command]()
