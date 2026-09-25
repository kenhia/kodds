# 004 — consumer-ready: per-call request log + consumer contract

korg: proposal 3228, slice 1 of program 3230 ("kodds 004: first consumer,
kmon severity as a shadow second reader"). It covers #3225 (the request log)
and #3226 (the consumer doc). It ran as an overseen karc leg on kubs0.
Slice 2 is kmon's second-reader step, korg:3229.

## Goal

Make kodds consumer-ready before kmon calls it. Every classify gets an id
and a log line, so a grade can later be joined to what Ken did with the
finding. And one page tells a consumer how to call kodds and what shape to
send.

## Premise check (sprint start)

- **#3225 held.** Nothing in `service.py` carried an id or wrote a log.
- **#3226 drifted, in the direction the proposal predicted.** Severity's 45
  eval items are single sentences (at most 114 characters, no newlines),
  split 15/15/15 across the three labels. Real kmon findings
  (homelab-health, and `korg_sink.py` on kai) have titles in that shape.
  Their details, though, are multi-line: a fenced journal tail for a failed
  unit, or several paragraphs of remedy text for storage and firmware
  findings. And kmon only files what failed. This was a proceed, not a
  refit: the shadow run measures this drift. It is documented in
  `docs/consumers.md`.

## Request log (#3225)

`kodds.calllog`: `CallLog(dir).append(record)` writes one compact JSON line
to `<dir>/<YYYY-MM>.jsonl` (UTC month). The file is created mode 600, in a
mode-700 dir, with `O_APPEND`, under a lock.
`Service.classify(task, inputs, caller=None)` stamps a `request_id` and
logs. The HTTP and MCP surfaces both go through it, so one code path covers
both.

- **Id: a ULID**, generated inline (48-bit ms time + 80 random bits,
  Crockford base32). Python 3.13 has no `uuid7`, and the format is not
  worth a dependency. The ids sort by time, so grep and sort order agree
  with file order.
- **Line fields:** ts, request_id, task, inputs, probs, **raw**, top,
  calibrated, model, prompt_sha256, latency_ms, caller. `raw` goes beyond
  the item's list on purpose. A later refit on shadow data needs the
  uncalibrated distribution, and without `raw` in the log it would have to
  re-score every input.
- **caller:** the body field or MCP tool argument, else the
  `X-Homelab-Agent` header, 1–64 characters. Anything else is a 422, and
  nothing is logged.
- **Never fails a call.** Serialisation and I/O errors are caught, warned
  once on stderr (the journal), and re-armed by the next successful write.
  The first test run caught a real bug here: `json.dumps` was outside the
  `try`, so an unserialisable record would have raised.
- Not logged: `/v1/score`, which has no task or calibration, and failed
  requests (404/422).
- `/healthz` gains `call_log: {path, writable}`, and `just deploy` prints
  it (with a warning if it is not writable). The unit sets
  `KODDS_CALLS_DIR` explicitly, beside `KODDS_PRIVATE_DIR`. The
  deploy-kodds skill's verify step now checks the last id landed, and the
  file and dir modes.

Tests (model-free): the id and the logged line agree over HTTP and MCP;
modes 600/700; caller comes from the body, the header, and the body over
the header; a bad caller is a 422 with no line; errors and score are not
logged; a log whose parent is a file serves every call and warns exactly
once; healthz reports the log. Plus unit tests for the ULID's shape and
order, the month file, one line per record, and the warning re-arming.

## Consumer contract (#3226)

`docs/consumers.md`, linked from the README. It covers the endpoints, the
three rules (threshold only on `calibrated`, pass `caller`, keep
`request_id`), timeouts and failing soft, what unavailable looks like, and
severity's input for a kmon `Finding`: `f"{title}\n{detail}"[:2000]`, the
task's own cap.

Severity latency, measured against the live 14B on first calls with unseen
inputs: ~115 ms for a one-liner, ~190 ms at 650 characters (a real
unit-failure finding), ~430 ms at the 2000-character cap. The 93 ms in 003
was a short, warm input. These are the numbers the doc quotes.

The "503 during restart" in the item was not grounded: kodds never emits a
5xx for a well-formed request. The doc therefore says a connection error,
or any 5xx from `tailscale serve`.

## Roadmap

The 004 entry now describes the program (two slices, shadow mode, gating
deferred) instead of "tentative".

## Cross-repo changes made

- **k-homelab `b55524a`** (pushed; `main...origin/main`, not ahead): a
  `facts/backup-coverage.md` row classifying the calls dir as **`GAP`**,
  the same way sprint 080 classified the route overlay. kubs0 has no
  path-list backup job. `just check-docs` is green.

## Follow-ups

- `recipes/kodds-service` in k-homelab asserts the overlay's mode, but not
  the calls dir's. Adding that would be new behaviour in another repo's
  recipe, so it is raised in the handoff, not landed here.
