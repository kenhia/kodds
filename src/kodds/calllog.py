"""The per-call request log: one JSON line per ``classify``, for joining later.

A consumer keeps the ``request_id`` beside whatever it files; when Ken later
disposes of that finding, the disposition can be joined back to kodds' grade
here. That join is the only source of real labels at kmon's volume, which is
why every classify is logged and nothing else is (``score`` has no task and no
calibration to judge).

Lines go to ``<dir>/<YYYY-MM>.jsonl`` (UTC month), mode 600 in a mode-700
directory: inputs are private korg content (route) or homelab findings. There
is no read API, no pruning and no label tool — the volume is a few lines a
week.

Logging never fails a classify. A write error is reported once on stderr (the
journal, under systemd) and the call is served anyway; the next successful
write re-arms the warning.
"""

import json
import os
import secrets
import sys
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def ulid(now_ms: int | None = None) -> str:
    """A ULID: 48-bit millisecond time + 80 random bits, Crockford base32.

    26 characters, lexically sortable by time, so a log grepped by id and a
    log sorted by id agree with the file's own order.
    """
    ms = int(time.time() * 1000) if now_ms is None else now_ms
    n = (ms << 80) | secrets.randbits(80)
    return "".join(CROCKFORD[(n >> (5 * i)) & 31] for i in reversed(range(26)))


class CallLog:
    """Appends call records under ``directory``; thread-safe, never raises."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._lock = threading.Lock()
        self._warned = False

    def path(self, when: datetime | None = None) -> Path:
        when = when or datetime.now(UTC)
        return self.directory / f"{when:%Y-%m}.jsonl"

    def append(self, record: Mapping[str, Any]) -> bool:
        """Write one line; False (after warning once) if it could not be written."""
        try:
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            line += "\n"
            with self._lock:
                self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                fd = os.open(self.path(), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                try:
                    os.write(fd, line.encode())
                finally:
                    os.close(fd)
        except (OSError, TypeError, ValueError) as e:
            if not self._warned:
                self._warned = True
                print(
                    f"kodds: call log write to {self.directory} failed ({e}); "
                    "serving results without logging them",
                    file=sys.stderr,
                )
            return False
        self._warned = False
        return True

    def status(self) -> dict[str, Any]:
        """For /healthz: where lines go, and whether they can be written there."""
        d = self.directory
        if d.exists():
            writable = d.is_dir() and os.access(d, os.W_OK | os.X_OK)
        else:
            parent = next((p for p in d.parents if p.exists()), None)
            writable = parent is not None and os.access(parent, os.W_OK | os.X_OK)
        return {"path": str(self.path()), "writable": writable}
