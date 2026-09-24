"""Build evals/evalset.jsonl — the labelled set the model bake-off runs on.

Three homelab-shaped tasks, each line ``{task, prompt, choices, label}``:

- ``route``    — which korg project owns a work item. Ground truth is the
  project real korg work items were filed under (titles copied verbatim,
  2026-09-24); the candidate projects' routing contracts are korg's own
  one-line descriptions.
- ``severity`` — kmon's three-way status call (ok / attention / problem) on a
  single finding. Findings are written in kmon's shape and labelled by hand
  against the rubric in kmon/controller.py ("THE THREE-WAY CALL").
- ``triage``   — a generic message triage: legitimate / spam / phishing.
  Synthetic messages, labelled by construction.

Run: ``uv run python evals/build_evalset.py``. The JSONL is committed so the
bake-off never depends on this script's environment; rerun it only when the
source lists below change.
"""

import json
from pathlib import Path

OUT = Path(__file__).with_name("evalset.jsonl")

# --- route -----------------------------------------------------------------

PROJECTS = {
    "klams": "Shared cross-agent memory: storage, retrieval, attribution, trust, "
    "the MCP surface.",
    "korg": "The work-tracking system itself: data model, MCP surface, server, web UI.",
    "kaed": "Ken's Agent Editor: agent-only file editing over HTTP MCP — contract, "
    "Rust daemon, per-host deploys, secrets handling.",
    "k-homelab": "Changing homelab machine state: installing software, system "
    "config, fleet recipes, audit, disaster recovery.",
    "kmon": "Scheduled read-only homelab monitoring: collect, synthesize, verify, "
    "report. kmon's own development work lands here.",
    "karc": "Ken's Agent Remote Control: launch, tail, steer and gate headless "
    "Claude Code sprint legs on other hosts.",
    "kvllm": "Serving local models on kai's 5090 via vLLM, plus the eval suites, "
    "judge and leaderboard.",
    "kpidash": "The Pi5 KPI dashboard: LVGL renderer, Redis contracts, client "
    "cards and panels.",
}

ROUTE_ITEMS = {
    "klams": [
        "[P1] Oversized chunks silently dropped — pass API 8192 check, then TEI 413 at the worker",
        "Recall quality: hand-authored gotchas lose to scanner-ingested code, incl. from deprecated repos",
        "Rework the authorization model: explicit trust grants, token-implied authorship, close the register_author backdoor",
        "REST routes enforce no scopes — a read-only token can delete knowledge and promote dissents",
        "Markdown chunker is fence-unaware — # comments inside code blocks become headings, shedding content-free chunks",
        'Scanner records repo as the scan-root basename — 218k of 222k points say repo="src"',
        "Query-time dedupe by content_hash + expose content_hash/author.id/heading_path in the search projection",
        "Weighted fusion: provenance (curated vs bulk) and declared-volatility weighting in ranking — eval-gated",
    ],
    "korg": [
        "Shared fetch/error surface in the web UI — no silent failures",
        "Modal focus management, labeled inputs, and proper row activation",
        "Fix fresh-install node sequence so node/WI #1 can exist",
        "Sprint proposals: display and filter by project (list_proposals + Planning page)",
        "Accept project by name wherever project_id is accepted, with an actionable unknown-name error",
        "MCP server instructions claim an envelope five collection reads don't return",
        "Ability to attach screen captures to WI's",
        "Clicking on background or other area should close filter dropdowns",
    ],
    "kaed": [
        "Version stamp: `--version` says 0.1.0 on every build, so no recipe can assert freshness",
        "Fleet deployed-ness is not discoverable from the client: a deferred host looks identical to a failed deploy",
        "Host-qualify root names and make the declared fleet part of the `roots` response",
        "`feedback` tool, re-shaped: fire at the moment of friction, not as a standing invitation",
        "Secrets slice 1: `.kaedignore`, in-file marker, structured refusal reasons, gitignore warning",
        "History tools: `journal`, `diff`, `revert`, plus a read path for `txn_failures`",
        "Gateway peer mode: proxy to peers with identity propagation and fleet-wide search",
        "Secrets slice 4: write-side leak detection and the cross-file occurrence index",
    ],
    "k-homelab": [
        "Fold in: klams-service backup.conf systemd drop-in on kubs0",
        "Prometheus/Grafana image refresh on kubsdb",
        "Fold in: OpenSearch decommissioned on kubs0",
        "Rotate the kubsdb service passwords (postgres, mongo, redis) + rpi53 redis",
        "ksandbox first recipes: incus substrate (incus + vmlab pool + incus-docker-fw), thin baseline, dr/ksandbox.md",
        "Install Go from a pinned upstream release on two homelab machines",
        "Decommission /krag NVMe on kubs0 — krag inactive, 1.8T disk idle",
        "dr/kubsdb.md step 8 restores korg's database but never recreates the `korg` role",
    ],
    "kmon": [
        "Weekly k-homelab drift report — its own korg report source, not a section of the daily",
        "Harden scheduled tools: blocking errors must file a korg report, not die silently",
        "Investigation probes ignore systemd scope — user units read as deleted, and our own timer is one",
        "verify.py false positive traps the monitor in a self-sustaining failure loop under systemd",
        "Unit-level suppression + expiry for suppressions.toml",
        "Add cross-mount storage-balance check (data on the wrong volume)",
        "A same-day re-run overwrites the previous report — keep the superseded run, don't discard it",
        "Overwatch v1: compare against yesterday's bundle, cross-reference listeners and timers against the k-homelab manifest",
    ],
    "karc": [
        "Spike: one systemd user unit per turn — launch, resume, and outlive the ssh session",
        "Ship-gate hook with Bash and Skill matchers, plus what headless mode does with AskUserQuestion",
        "Lock module: kaed-path prefix conflicts, TOML lock file at .git/karc.lock, startup reconcile — with tests",
        "`tail`: the event projection and its cursor — what the overseer reads instead of raw stream-json",
        "PD-6: what a leg is allowed to reach — decide the MCP toolbox for unattended legs",
        "`send`: a resume turn on an idle or asked leg — refused while a turn runs, banner-derived status afterwards",
        "`stop` and `stop --abandon`: end the current turn, terminal state, lock kept unless abandoned",
        "kubs0 instance + kai as gateway: host-qualified legs, peer tokens, `list`/`launch` proxied and journaled on the target",
    ],
    "kvllm": [
        "Serving ergonomics: model registry + serve recipes + quant notes + /v1 contract",
        "Availability: systemd unit + auto-restart on kai reboot",
        "Model collection research — best free coding/agentic models that fit a 5090",
        "Eval harness v2 (Inspect AI, sandboxed agentic+coding, weighted leaderboard)",
        "Vision v2: classification, captioning, render-QA",
        "Bump vLLM 0.24.0 → 0.26.x for Gated DeltaNet support",
        "Land a qwen3.8-27b registry entry that actually serves on the 5090",
        "Establish the eval suite's noise floor (N repeated runs) before reading any leaderboard gap",
    ],
    "kpidash": [
        'Add nerdfont icon `f0a07` and use it for the rpidash "self service card"',
        'klam\'s service card shows "unreachable" at times when `klams` is up',
        "Service Card: kmon status with staleness alerting",
        "Dashboard burns ~1.6 cores on rpi53: per-task pthread handshake × 8,100 chart line segments/sec",
        '"<host> stale" indicator in the service row for any homelab host except rpi53',
        "A service that has never published renders as no card, not as RED",
        "Service Card text: a character outside the panel font floods the journal, and the card contract never says which characters are safe",
        'A host that is down across a dashboard restart shows no card — admission-on-data has no durable "has published" marker',
    ],
}


def route_prompt(title: str) -> str:
    lines = "\n".join(f"- {name}: {desc}" for name, desc in PROJECTS.items())
    return (
        "You route work items to the homelab project that owns them.\n\n"
        f"Projects:\n{lines}\n\n"
        f"Work item: {title}\n\n"
        "Answer with the project name only."
    )


# --- severity --------------------------------------------------------------

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


def severity_prompt(finding: str) -> str:
    return (
        "You grade homelab monitoring findings. The status is what Ken does with "
        "the report:\n"
        "- ok: nothing needs doing, or the finding only needs recording.\n"
        "- attention: a human needs to decide or act, but not urgently.\n"
        "- problem: escalate now; Ken gets woken up for this.\n\n"
        f"Finding: {finding}\n\n"
        "Answer with ok, attention or problem only."
    )


# --- triage ----------------------------------------------------------------

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


def triage_prompt(message: str) -> str:
    return (
        "You triage incoming email.\n"
        "- legitimate: real mail the recipient expects or would want.\n"
        "- spam: unsolicited bulk advertising or junk, not trying to steal anything.\n"
        "- phishing: tries to steal credentials, money or secrets by impersonation.\n\n"
        f"Message: {message}\n\n"
        "Answer with legitimate, spam or phishing only."
    )


def rows():
    route_choices = list(PROJECTS)
    for label, titles in ROUTE_ITEMS.items():
        for title in titles:
            yield {
                "task": "route",
                "prompt": route_prompt(title),
                "choices": route_choices,
                "label": label,
            }
    for label, findings in SEVERITY_ITEMS.items():
        for finding in findings:
            yield {
                "task": "severity",
                "prompt": severity_prompt(finding),
                "choices": ["ok", "attention", "problem"],
                "label": label,
            }
    for label, messages in TRIAGE_ITEMS.items():
        for message in messages:
            yield {
                "task": "triage",
                "prompt": triage_prompt(message),
                "choices": ["legitimate", "spam", "phishing"],
                "label": label,
            }


def main() -> None:
    with OUT.open("w") as f:
        for row in rows():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
