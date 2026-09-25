# 004b — fast restart under tailnet traffic

korg: proposal 3253, the slice the overseer inserted at rank 0.5 in program
3230, after slice 1's ship turn found the hang (handoff korg:3251). It covers
#3248 and ran as an overseen karc leg on kubs0.

## Goal

A deploy restart under live tailnet traffic showed 91 s of 502. uvicorn
logged `Waiting for connections to close` and sat there until systemd's
default 90 s stop timeout SIGKILLed it. Bring that window under 10 s.

## Premise check (sprint start)

**#3248 held.** `service.main` called `uvicorn.run(app, host=, port=)` with
no graceful-shutdown bound, and `deploy/kodds.service` had no
`TimeoutStopSec`.

## What changed

- `service.serve(app, host, port)` wraps `uvicorn.run` with
  `timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S` (5). `main` calls it. It
  exists as a function so a test can check the argument without loading a
  model.
- `deploy/kodds.service` sets `TimeoutStopSec=15`. uvicorn's own 5 s bound
  should always fire first. This is the backstop, and it is still well short
  of the 90 s default.
- `docs/consumers.md` now says under 10 s of 502 instead of about 90 s. The
  measured figure is added at the deploy (below).
- Tests: `serve` passes the bound, the bound is at most 10 s, and the unit's
  `TimeoutStopSec` sits between the bound and 90.

## What was actually holding the stop open

Correction to #3248's diagnosis, found while proving the fix without a GPU
(`.scratch/stream_stop.py`, `.scratch/keepalive_stop.py`, a stub Starlette
app served through `serve`):

- An **idle** keep-alive connection does *not* delay the stop. uvicorn
  closes those at once: the stop took 0.2 s with one open.
- An **open streaming response** does. Unbounded, the stub printed exactly
  the production line, `Waiting for connections to close`, and was still
  running 20 s later. With the 5 s bound, it logged `Cancel 1 running
  task(s), timeout graceful shutdown exceeded` and exited in **5.2 s**.

So the 004 hang was an in-flight or long-lived request through `tailscale
serve`, not the proxy's idle pool. The fix is the same either way. (The
deploy then ruled out the obvious long-lived candidate, an MCP GET stream:
the server ends those itself when shutdown starts. See below.)

## Acceptance (at the deploy)

This is a trigger, judged in the ship turn. `.scratch/restart_probe.sh`
polls `https://kubs0.encke-wahoo.ts.net:7780/healthz` every 0.2 s,
bounded by `timeout 180`. It runs `just deploy` in the same foreground
script, waits for the poller, reports the non-200 window from epoch
stamps, and greps the journal for the stop lines. Pass: a window under
10 s, and no `stop-sigterm timed out`.

## Deployed

- **What:** `95c16c8` (squash of PR #5), by the `deploy-kodds` skill
  (`just deploy` inside `.scratch/restart_probe.sh`), on 2026-09-25 at
  12:15 PDT, run from kubs0. `current` → `releases/95c16c8…`, and
  `previous` → `5162bf4`.
- **healthz** (over the tailnet): the commit matches, `gpu_offload` is
  true, and route, severity and triage are all `calibrated=True`. The call
  log is writable, with modes 700 (dir) and 600 (file).
- **Tailnet classify:** severity → attention (0.11 s), triage → phishing
  (0.12 s), route → krot (8.3 s, the first route call after a restart).
  All three are calibrated and carry request ids. The last id
  (`01M3D00DWM26XS93WMD2K5FB78`) is the last line of the call log. MCP
  `tools/list` returns list_tasks, classify and score, and an MCP
  `classify` came back calibrated with a request id.
- **VRAM:** 13,402 MiB before, 13,404 MiB after. **TEI pids** were 875202
  and 875203 before and after, untouched.

### Restart acceptance: three restarts, all from kubs0 over the tailnet

The poller hit `https://kubs0.encke-wahoo.ts.net:7780/healthz` every 0.2 s
and stamped each result with epoch seconds. Everything ran in the
foreground and bounded.

| # | what stopped | held through the proxy | stop | 502 window |
|---|---|---|---|---|
| 1 | old `5162bf4` (unbounded) under the new unit | MCP GET stream | `stop-sigterm timed out` at **15 s**, SIGKILL (the backstop) | 23.3 s |
| 2 | new `95c16c8` | MCP GET stream | clean, `Finished server process`, under 1 s | **2.9 s** |
| 3 | new `95c16c8` | 24 queued route classifies | `Cancel 24 running task(s), timeout graceful shutdown exceeded` at **+5 s**, clean `Stopped` | **7.5 s** |

- **Run 1 could not test the fix.** The process being stopped was the
  old release. It does show the unit backstop working live: 15 s where
  004 took 90 s.
- **The MCP GET stream is not a holding connection.** In both runs it
  returned 200, then logged `Terminating session` at the moment shutdown
  began. The server closes those streams itself. So what held the old
  process in run 1 (and for 91 s at 004) was some other connection,
  which is still unidentified. The bound covers it whatever it is.
- **Run 3 is the proof the overseer asked for** (comment 3108): real
  in-flight work, held longer than 5 s through the proxy. The bound fired
  and the stop was clean.
- **Finding, fixed in the doc:** the 24 cancelled requests got **HTTP 500
  from kodds**, with no request id and no call-log line. `consumers.md`
  had said kodds never returns a 5xx for a well-formed request. It now
  names this case, and the advice is the same: kodds is unavailable, so
  retry at most once.
