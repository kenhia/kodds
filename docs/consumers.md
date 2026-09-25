# Calling kodds: the consumer contract

This is the page a consumer repo reads before it calls kodds. kmon is the
first consumer: it grades each finding with `severity` as a shadow second
reader (program korg:3230). The rules are the same for any consumer.

## Endpoints

One service on kubs0, published to the tailnet by `tailscale serve`:

| surface | call |
|---|---|
| HTTP | `POST https://kubs0.encke-wahoo.ts.net:7780/v1/classify` with `{task, inputs, caller?}` |
| MCP (streamable HTTP, stateless) | `https://kubs0.encke-wahoo.ts.net:7780/mcp`, tool `classify(task, inputs, caller?)` |
| task list | `GET /v1/tasks`, or MCP `list_tasks`: each task's input names, choices and `calibrated` |

It is **tailnet-only with no token**. Don't publish it further.

```sh
curl -s https://kubs0.encke-wahoo.ts.net:7780/v1/classify \
  -H 'content-type: application/json' -H 'X-Homelab-Agent: kmon' \
  -d '{"task":"severity","inputs":{"finding":"kubsdb: root fs at 97%"},"caller":"kmon"}'
```

A response:

```json
{"request_id": "01K...", "task": "severity",
 "probs": {"ok": 0.01, "attention": 0.91, "problem": 0.08},
 "raw": {...}, "calibrated": true, "model": "Qwen3-14B-Q4_K_M.gguf",
 "top": "attention", "latency_ms": 118.9, "queued_ms": 0.0}
```

`/v1/score` (ad-hoc prompt and choices) exists but is always raw and is not
logged. A consumer that wants a decision it can threshold uses `classify`.

## The three rules

1. **Threshold only on `calibrated: true`.** When it is true, `probs` has been
   temperature-scaled on a held-out set (sprint 002), and a 0.8 means roughly
   0.8. When it is false, `probs` equals `raw`, and raw is badly
   overconfident: a measured severity call read 0.998 raw and 0.77
   calibrated. A task goes uncalibrated when its prompt or model no longer
   matches the fit. Treat that as "rank with it, don't gate on it", and
   surface it rather than hiding it. `/healthz` shows per-task `calibrated`
   and `prompt_matches_fit`.
2. **Pass `caller`**: your service or agent name, 1–64 characters. Use the
   `caller` body field (or tool argument), or the `X-Homelab-Agent` header.
   The body field wins when both are sent. kmon sends `kmon`.
3. **Keep `request_id` beside whatever you file.** Every classify is logged
   on kubs0, one JSON line per call, in
   `~/.local/share/kodds/calls/<YYYY-MM>.jsonl` (UTC month, mode 600). Each
   line holds the id, caller, task, inputs, `probs`, `raw`, `top`,
   `calibrated`, model and prompt hash. When Ken later decides what a
   finding really was, the id is what joins that decision to kodds' grade.
   Without it, the grade can never become a label. There is no read API; the
   join is done offline, on kubs0.

## Timeouts, and failing soft

Inference is serialised: llama.cpp is not re-entrant, so concurrent callers
queue (`queued_ms`). Measured on the resident 14B:

| call | latency |
|---|---|
| severity, one-line finding | ~115 ms first call, ~50 ms repeated |
| severity, a kmon finding with a journal tail (~650 chars) | ~190 ms |
| severity at its 2000-char input cap | ~430 ms |
| route | ~1.45 s |
| cold start, or a route call straight after severity | ~5–9 s (the first route call after the 004 deploy took 8.7 s) |

**Use a 10 s client timeout and fail soft.** kodds is advice. A consumer
that cannot reach it records "no grade" and carries on exactly as it would
have without kodds. Nothing a consumer does may *depend* on a grade arriving.

### What unavailable looks like

- **A connection error, or an HTTP 502 from `tailscale serve`**: the
  process is not answering. Every deploy restarts the service. The old
  process gives in-flight requests at most 5 s to finish, then cancels
  them, and the model loads in about 2 s. So **a restart under live
  tailnet traffic shows under 10 s of 502**: measured 2.9 s with nothing
  in flight and 7.5 s with 24 route calls queued (004b deploy). Before
  004b it waited on them until systemd killed it, about 90 s (kodds
  #3248). A refit stops the service on purpose for about 15 minutes (see
  `route-overlay.md`).
- **An HTTP 500 from kodds itself**: your request was in flight when a
  restart cancelled it. It has no `request_id` and no line in the call
  log. Treat it like the 502: kodds is unavailable, not your bug. Outside
  a restart, kodds does not return a 5xx for a well-formed request.
- **A timeout**: a long queue behind route calls, or a stuck process.
- `404` means an unknown task. `422` means bad inputs or a bad `caller`.
  `400` means the body was not a JSON object. These are your bug, not
  kodds' state. Don't retry them.

**Don't retry in a loop.** At most one retry after a few seconds. After that,
record the miss and move on. kmon runs once a night, so a grade that is
missing for one night costs nothing. A retry loop against a restarting
service costs a GPU queue that other callers are waiting in.

## `severity`: the input for a kmon finding

`severity` takes one input, `finding`. Its choices are `ok` (nothing to do,
or record only), `attention` (a human decides or acts, not urgently) and
`problem` (escalate now; Ken gets woken). kmon's `Finding`
(`kmon/korg_sink.py`) has a `title` and a `detail`. Send them as:

```python
finding = f"{f.title}\n{f.detail}"[:2000]
```

That is the title, one newline, then the detail, cut at 2,000 characters.
2,000 is the task's own input cap (`tasks/severity.toml`, `[limits]`).
kodds would cut there anyway and append `…`. Cutting on the consumer's side
makes the cut visible in the consumer's code. Keep the title first: it
carries the host, the unit or mount, and the state, which is most of the
signal.

### How real findings differ from what severity was calibrated on

Severity's calibration was fitted on 45 hand-written items "in kmon's shape"
(`evals/evalset/severity.*.jsonl`). Compared with real kmon findings
(homelab-health work items, and `korg_sink.py`'s builders), those items
differ in two material ways. The shadow run is exactly what measures this
drift, so this sprint does **not** refit.

1. **Length and structure.** Every eval item is one sentence, at most 114
   characters, with no newlines. Real titles match that shape
   (`kubs0: xdg-desktop-portal-gtk.service failing (failed/failed)`), but a
   real detail is multi-line and much longer. A failed unit's detail is
   `Observed by kmon.` plus a fenced journal tail, about 650 characters
   together with its title. A storage or firmware-NVRAM finding's detail
   runs to several paragraphs, including remedy text that the eval items
   never contain. Spot checks put these in the class you would expect: a
   one-line eval-style input and a real unit failure both came out
   `attention`, at 0.91 and 0.94. But the calibration was never fitted on
   inputs like these, so its reliability on them is unmeasured.
2. **Base rate.** The eval set is balanced at 15 `ok`, 15 `attention` and
   15 `problem`. kmon files a finding only when something failed or crossed
   a threshold, so a genuine `ok` should be rare in real traffic. Ken's
   dispositions, though, show many filed findings turning out benign (the
   `run-u*.service` transients, a GUI portal on a headless host). Whether
   those should read as `ok` or `attention` is the disagreement the shadow
   run exists to record.

Read the shadow run's disagreements with both of these in mind before
treating them as model error.

## Other tasks

`route` (which korg project owns a work item: `title`, `content`) and
`triage` (`message`) follow the same rules. Route runs calibrated only while
its pinned private overlay matches the fit; see
[route-overlay.md](route-overlay.md). Route is not a consumer-facing
decision in program 004: its top-1 accuracy is 64.7%.
