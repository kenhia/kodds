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
serve`, most likely an MCP streamable-HTTP stream, not the proxy's idle
pool. The fix is the same either way.

## Acceptance (at the deploy)

This is a trigger, judged in the ship turn. `.scratch/restart_probe.sh`
polls `https://kubs0.encke-wahoo.ts.net:7780/healthz` every 0.2 s,
bounded by `timeout 180`. It runs `just deploy` in the same foreground
script, waits for the poller, reports the non-200 window from epoch
stamps, and greps the journal for the stop lines. Pass: a window under
10 s, and no `stop-sigterm timed out`.
