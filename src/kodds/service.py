"""The kodds service: one resident model, served over HTTP and MCP.

One process loads the model once and keeps it. HTTP (``/healthz``, ``/v1/*``)
and MCP (``/mcp``, streamable HTTP) are two views of the same ``Service``,
so there is one model in VRAM, not one per surface.

llama.cpp is not re-entrant, so every inference runs behind one lock:
concurrent callers queue. Inference runs in a worker thread, which keeps the
event loop — and ``/healthz`` — answering while a long route call holds the
lock.

The service binds loopback only; ``tailscale serve --https=7780`` publishes it
to the tailnet (the fleet pattern — binding the tailnet address or 0.0.0.0
collides with tailscaled's own listener on the next restart).

``main`` refuses to start on a CPU-only llama.cpp build or without the VRAM
the model needs: a CPU build would serve at ~100x the latency, and loading
into too little VRAM would fight klams' TEI embedders for the card.
"""

import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn

import anyio.to_thread
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse

from kodds.scorer import Scorer
from kodds.tasks import TASKS_DIR, Task, load_tasks, overlay_path

ROOT = TASKS_DIR.parent
PORT = 7780
LOOPBACK = ("127.0.0.1", "localhost", "::1")
# Host headers the MCP transport accepts (DNS-rebinding protection). tailscale
# serve passes the ts.net name through, so it has to be listed.
DEFAULT_PUBLIC_HOSTS = ("kubs0.encke-wahoo.ts.net",)

RAW_WARNING = (
    "Raw probabilities (calibrated=false) are overconfident: 002 fitted "
    "temperatures of about 5 for severity/triage. Threshold only on results "
    "with calibrated=true."
)


class UnknownTask(KeyError):
    pass


class BadRequest(ValueError):
    pass


@dataclass
class Service:
    """The model and tasks the service serves; blocking, model-agnostic."""

    scorer: Scorer
    tasks: Mapping[str, Task]
    commit: str | None = None
    model_path: str | None = None
    private_dir: Path | None = None
    gpu_offload: bool | None = None
    vram: Callable[[], dict[str, int] | None] = lambda: None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def task(self, name: str) -> Task:
        try:
            return self.tasks[name]
        except KeyError:
            raise UnknownTask(name) from None

    def task_info(self, task: Task) -> dict[str, Any]:
        fitted = task.calibrations.get(self.scorer.model or "")
        prompt = task.prompt_hash()
        return {
            "name": task.name,
            "inputs": task.inputs,
            "choices": list(task.choices),
            "calibrated": task.calibration_for(self.scorer.model) is not None,
            "prompt_sha256": prompt,
            "fitted_prompt_sha256": fitted.prompt_sha256 if fitted else None,
            "prompt_matches_fit": bool(fitted) and fitted.prompt_sha256 == prompt,
            "overlay": overlay_path(TASKS_DIR / task.name, self.private_dir).exists(),
        }

    def list_tasks(self) -> list[dict[str, Any]]:
        return [self.task_info(t) for t in self.tasks.values()]

    def classify(self, task: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
        t = self.task(task)
        if not isinstance(inputs, Mapping) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in inputs.items()
        ):
            raise BadRequest("inputs must be an object of strings")
        wanted = t.inputs
        missing = [n for n in wanted if n not in inputs]
        extra = [n for n in inputs if n not in wanted]
        if missing or extra:
            raise BadRequest(
                f"task {task!r} takes inputs {wanted}: "
                f"missing {missing}, unknown {extra}"
            )
        result, timing = self._run(lambda: t.classify(self.scorer, **inputs))
        return {
            "task": t.name,
            "probs": result.probs,
            "raw": result.raw,
            "calibrated": result.calibrated,
            "model": result.model,
            "top": max(result.probs, key=result.probs.__getitem__),
            **timing,
        }

    def score(
        self, prompt: str, choices: Sequence[str], system: str | None = None
    ) -> dict[str, Any]:
        if not isinstance(prompt, str) or not prompt:
            raise BadRequest("prompt must be a non-empty string")
        if (
            isinstance(choices, str)
            or not isinstance(choices, Sequence)
            or not choices
            or not all(isinstance(c, str) and c for c in choices)
        ):
            raise BadRequest("choices must be a non-empty list of non-empty strings")
        if len(set(choices)) != len(choices):
            raise BadRequest("choices must be distinct")
        if system is not None and not isinstance(system, str):
            raise BadRequest("system must be a string")
        probs, timing = self._run(
            lambda: self.scorer.score(prompt, list(choices), system)
        )
        return {
            "probs": probs,
            "raw": probs,
            "calibrated": False,
            "model": self.scorer.model,
            "top": max(probs, key=probs.__getitem__),
            **timing,
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "loaded": True,
            "model": self.scorer.model,
            "model_path": self.model_path,
            "gpu_offload": self.gpu_offload,
            "vram_mib": self.vram(),
            "commit": self.commit,
            "private_dir": str(self.private_dir) if self.private_dir else None,
            "tasks": {t["name"]: t for t in self.list_tasks()},
        }

    def _run[T](self, fn: Callable[[], T]) -> tuple[T, dict[str, float]]:
        """``fn()`` under the model lock, with queue and inference times."""
        asked = time.perf_counter()
        with self._lock:
            started = time.perf_counter()
            value = fn()
            done = time.perf_counter()
        return value, {
            "latency_ms": round((done - started) * 1000, 1),
            "queued_ms": round((started - asked) * 1000, 1),
        }


def build_mcp(service: Service) -> MCPServer:
    mcp = MCPServer(
        "kodds",
        instructions=(
            "Local LLM classifier on kubs0: per-choice probabilities read from "
            "a model's logits, no generation. Prefer `classify` with a named "
            "task (calibrated); `score` is for ad-hoc choice sets and is always "
            "raw. " + RAW_WARNING
        ),
    )

    read_only = ToolAnnotations(read_only_hint=True, idempotent_hint=True)

    @mcp.tool(
        description="The named tasks: their inputs, choices, and whether each is "
        "calibrated for the loaded model.",
        annotations=read_only,
    )
    async def list_tasks() -> list[dict[str, Any]]:
        return service.list_tasks()

    @mcp.tool(
        description="Classify with a named task (see list_tasks): probabilities "
        "over its fixed choices, with `top` the most likely. `probs` is "
        "calibrated when `calibrated` is true and otherwise equals `raw`. "
        + RAW_WARNING,
        annotations=read_only,
    )
    async def classify(task: str, inputs: dict[str, str]) -> dict[str, Any]:
        return await anyio.to_thread.run_sync(service.classify, task, inputs)

    @mcp.tool(
        description="Probability of each choice as the answer to `prompt`, "
        "normalised over `choices`; choices are scored as whole strings, so "
        "they may span several tokens. Always raw (`calibrated` false): use it "
        "to rank, not to threshold. " + RAW_WARNING,
        annotations=read_only,
    )
    async def score(
        prompt: str, choices: list[str], system: str | None = None
    ) -> dict[str, Any]:
        return await anyio.to_thread.run_sync(service.score, prompt, choices, system)

    return mcp


def create_app(
    service: Service, *, public_hosts: Sequence[str] = DEFAULT_PUBLIC_HOSTS
) -> Starlette:
    """HTTP routes and the MCP endpoint on one Starlette app."""
    mcp = build_mcp(service)

    def error(status: int, message: str) -> JSONResponse:
        return JSONResponse({"error": message}, status_code=status)

    async def body(request: Request) -> dict[str, Any]:
        try:
            data = await request.json()
        except ValueError:
            raise BadRequest("body must be JSON") from None
        if not isinstance(data, dict):
            raise BadRequest("body must be a JSON object")
        return data

    async def call(fn: Callable[[], Any]) -> JSONResponse:
        try:
            return JSONResponse(await anyio.to_thread.run_sync(fn))
        except UnknownTask as e:
            return error(404, f"unknown task {e.args[0]!r}; see GET /v1/tasks")
        except BadRequest as e:
            return error(422, str(e))

    @mcp.custom_route("/healthz", methods=["GET"])
    async def healthz(request: Request) -> JSONResponse:
        return JSONResponse(await anyio.to_thread.run_sync(service.health))

    @mcp.custom_route("/v1/tasks", methods=["GET"])
    async def tasks(request: Request) -> JSONResponse:
        return JSONResponse(service.list_tasks())

    @mcp.custom_route("/v1/classify", methods=["POST"])
    async def classify(request: Request) -> JSONResponse:
        try:
            data = await body(request)
        except BadRequest as e:
            return error(400, str(e))
        if not isinstance(data.get("task"), str):
            return error(422, "task (string) is required")
        return await call(
            lambda: service.classify(data["task"], data.get("inputs", {}))
        )

    @mcp.custom_route("/v1/score", methods=["POST"])
    async def score(request: Request) -> JSONResponse:
        try:
            data = await body(request)
        except BadRequest as e:
            return error(400, str(e))
        return await call(
            lambda: service.score(
                data.get("prompt", ""), data.get("choices", []), data.get("system")
            )
        )

    hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
    for h in public_hosts:
        hosts += [h, f"{h}:*"]
    security = TransportSecuritySettings(
        allowed_hosts=hosts,
        allowed_origins=[f"https://{h}" for h in hosts if not h.endswith(":*")]
        + [f"http://{h}" for h in hosts],
    )
    # Stateless + JSON: a restart (every deploy) must not strand clients on a
    # session id the new process has never seen.
    return mcp.streamable_http_app(
        stateless_http=True, json_response=True, transport_security=security
    )


def free_vram_mib() -> dict[str, int] | None:
    """Free and total VRAM on GPU 0 per nvidia-smi, or None without one."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    free, total = (int(v) for v in out.splitlines()[0].split(","))
    return {"free": free, "total": total}


def git_commit(root: Path = ROOT) -> str | None:
    """The deployed commit: a release dir's ``COMMIT`` file, else git's HEAD."""
    stamp = root / "COMMIT"
    if stamp.exists():
        return stamp.read_text().strip() or None
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def fail(message: str) -> NoReturn:
    print(f"kodds: refusing to start: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    """Load the model and serve. Configured by environment:

    ``KODDS_MODEL`` (required) the GGUF; ``KODDS_N_CTX`` (8192);
    ``KODDS_HOST`` (127.0.0.1, loopback only); ``KODDS_PORT`` (7780);
    ``KODDS_PRIVATE_DIR`` where task overlays live (default ``tasks/private``);
    ``KODDS_MIN_FREE_MIB`` VRAM that must be free before loading (11000);
    ``KODDS_PUBLIC_HOSTS`` comma-separated Host names MCP accepts besides
    loopback (the kubs0 ts.net name).
    """
    import llama_cpp
    import uvicorn

    from kodds.llama import load

    env = os.environ
    if not env.get("KODDS_MODEL"):
        fail("KODDS_MODEL is not set")
    model_path = Path(env["KODDS_MODEL"]).expanduser()
    if not model_path.exists():
        fail(f"model file {model_path} does not exist")
    host = env.get("KODDS_HOST", "127.0.0.1")
    if host not in LOOPBACK:
        fail(f"KODDS_HOST={host}: bind loopback and publish with tailscale serve")
    port = int(env.get("KODDS_PORT", PORT))
    n_ctx = int(env.get("KODDS_N_CTX", 8192))
    private = env.get("KODDS_PRIVATE_DIR")
    private_dir = Path(private).expanduser() if private else None
    public_hosts = [
        h.strip()
        for h in env.get("KODDS_PUBLIC_HOSTS", ",".join(DEFAULT_PUBLIC_HOSTS)).split(
            ","
        )
        if h.strip()
    ]

    if not llama_cpp.llama_supports_gpu_offload():
        fail(
            "llama-cpp-python has no GPU offload (a CPU build). Rebuild it with "
            "CMAKE_ARGS=-DGGML_CUDA=on uv sync --reinstall-package llama-cpp-python"
        )
    need = int(env.get("KODDS_MIN_FREE_MIB", 11000))
    vram = free_vram_mib()
    if vram is None:
        fail("nvidia-smi found no GPU")
    elif vram["free"] < need:
        fail(
            f"{vram['free']} MiB VRAM free, {need} MiB needed; "
            "not loading beside what is already on the card"
        )

    t0 = time.perf_counter()
    scorer = load(model_path, n_ctx=n_ctx)
    tasks = load_tasks(private_dir=private_dir)
    service = Service(
        scorer,
        tasks,
        commit=git_commit(),
        model_path=str(model_path),
        private_dir=private_dir,
        gpu_offload=True,
        vram=free_vram_mib,
    )
    print(
        f"kodds: {scorer.model} loaded in {time.perf_counter() - t0:.1f}s "
        f"(n_ctx {n_ctx}), commit {service.commit}",
        file=sys.stderr,
    )
    for info in service.list_tasks():
        print(
            f"kodds: task {info['name']}: calibrated={info['calibrated']} "
            f"overlay={info['overlay']}",
            file=sys.stderr,
        )
    uvicorn.run(create_app(service, public_hosts=public_hosts), host=host, port=port)


if __name__ == "__main__":
    main()
