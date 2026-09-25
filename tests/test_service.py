import json
import stat
import threading
import time

import anyio
import pytest
from mcp import Client
from starlette.testclient import TestClient
from test_scorer import plain
from test_tasks import spam_model, spam_task

from kodds.calllog import CallLog
from kodds.scorer import Scorer
from kodds.service import (
    GRACEFUL_SHUTDOWN_S,
    ROOT,
    BadRequest,
    Service,
    UnknownTask,
    build_mcp,
    create_app,
    serve,
)
from kodds.tasks import Calibration

MODEL = "m.gguf"


def service(**kw) -> Service:
    scorer = Scorer(spam_model(), template=plain, model=MODEL)
    fitted = spam_task(template="{message}:")
    cal = Calibration(
        temperature=4.0, choices=("spam", "ham"), prompt_sha256=fitted.prompt_hash()
    )
    tasks = {
        "spam": spam_task(template="{message}:", calibrations={MODEL: cal}),
        "raw": spam_task(name="raw", template="{message}:"),
    }
    return Service(scorer, tasks, commit="abc123", **kw)


@pytest.fixture
def calls(tmp_path):
    return CallLog(tmp_path / "calls")


@pytest.fixture
def client(calls):
    with TestClient(create_app(service(calls=calls))) as c:
        yield c


def logged(calls: CallLog) -> list[dict]:
    path = calls.path()
    return (
        [json.loads(ln) for ln in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def test_classify_returns_calibrated_probs_and_raw(client):
    r = client.post("/v1/classify", json={"task": "spam", "inputs": {"message": ""}})
    assert r.status_code == 200
    body = r.json()
    assert body["calibrated"] is True
    assert body["model"] == MODEL
    assert body["top"] == "spam"
    # T=4 pulls the calibrated probability toward 0.5, below the raw one.
    assert 0.5 < body["probs"]["spam"] < body["raw"]["spam"]
    assert body["latency_ms"] >= 0 and body["queued_ms"] >= 0


def test_classify_without_a_fit_passes_raw_through(client):
    r = client.post("/v1/classify", json={"task": "raw", "inputs": {"message": ""}})
    body = r.json()
    assert body["calibrated"] is False
    assert body["probs"] == body["raw"]


def test_unknown_task_is_404(client):
    r = client.post("/v1/classify", json={"task": "nope", "inputs": {}})
    assert r.status_code == 404
    assert "nope" in r.json()["error"]


@pytest.mark.parametrize(
    "inputs",
    [{}, {"message": "x", "extra": "y"}, {"message": 3}, ["message"]],
)
def test_bad_inputs_are_422(client, inputs):
    r = client.post("/v1/classify", json={"task": "spam", "inputs": inputs})
    assert r.status_code == 422


def test_malformed_body_is_400(client):
    r = client.post("/v1/classify", content=b"not json")
    assert r.status_code == 400
    assert client.post("/v1/classify", json=["spam"]).status_code == 400


def test_score_is_always_raw(client):
    r = client.post("/v1/score", json={"prompt": ":", "choices": ["spam", "ham"]})
    assert r.status_code == 200
    body = r.json()
    assert body["calibrated"] is False
    assert body["top"] == "spam"
    assert body["probs"] == body["raw"]


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": ["a"]},
        {"prompt": ":", "choices": []},
        {"prompt": ":", "choices": "ab"},
        {"prompt": ":", "choices": ["a", "a"]},
        {"prompt": ":", "choices": ["a"], "system": 1},
    ],
)
def test_bad_score_requests_are_422(client, payload):
    assert client.post("/v1/score", json=payload).status_code == 422


def test_tasks_and_healthz_report_calibration(client):
    tasks = {t["name"]: t for t in client.get("/v1/tasks").json()}
    assert tasks["spam"]["calibrated"] and tasks["spam"]["prompt_matches_fit"]
    assert not tasks["raw"]["calibrated"]
    assert tasks["spam"]["inputs"] == ["message"]
    health = client.get("/healthz").json()
    assert health["loaded"] and health["model"] == MODEL
    assert health["commit"] == "abc123"
    assert health["tasks"]["spam"]["calibrated"]


def test_healthz_shows_a_drifted_prompt():
    svc = service()
    task = svc.tasks["spam"]
    # The prompt changed after the fit (e.g. a regenerated overlay).
    svc.tasks = {
        "spam": spam_task(template="{message}!:", calibrations=task.calibrations)
    }
    info = svc.health()["tasks"]["spam"]
    assert not info["calibrated"]
    assert not info["prompt_matches_fit"]
    assert info["fitted_prompt_sha256"] != info["prompt_sha256"]


def mcp_tools_list(client, host):
    return client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers={"Host": host, "Accept": "application/json, text/event-stream"},
    )


def test_mcp_rejects_foreign_host_headers(client):
    assert mcp_tools_list(client, "evil.example").status_code == 421


@pytest.mark.parametrize(
    "host",
    ["kubs0.encke-wahoo.ts.net:7780", "kubs0.encke-wahoo.ts.net", "127.0.0.1:7780"],
)
def test_mcp_over_http_on_loopback_and_tailnet_hosts(client, host):
    r = mcp_tools_list(client, host)
    assert r.status_code == 200
    names = {t["name"] for t in r.json()["result"]["tools"]}
    assert names == {"list_tasks", "classify", "score"}


class SlowBackend:
    """Wraps a backend, recording how many calls overlap."""

    def __init__(self, inner):
        self.inner = inner
        self.active = 0
        self.peak = 0
        self.guard = threading.Lock()

    def tokenize(self, text):
        return self.inner.tokenize(text)

    def continuation_logits(self, prefix, continuation):
        with self.guard:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.02)
        with self.guard:
            self.active -= 1
        return self.inner.continuation_logits(prefix, continuation)


def test_inference_is_serialised():
    backend = SlowBackend(spam_model())
    svc = service()
    svc.scorer = Scorer(backend, template=plain, model=MODEL)
    threads = [
        threading.Thread(target=svc.classify, args=("spam", {"message": ""}))
        for _ in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert backend.peak == 1


def test_service_errors():
    svc = service()
    with pytest.raises(UnknownTask):
        svc.classify("nope", {})
    with pytest.raises(BadRequest):
        svc.classify("spam", {"message": "x", "other": "y"})


def test_mcp_round_trip_in_process():
    async def run():
        async with Client(build_mcp(service())) as c:
            tools = {t.name: t for t in (await c.list_tools()).tools}
            result = await c.call_tool(
                "classify", {"task": "spam", "inputs": {"message": ""}}
            )
            return tools, result

    tools, result = anyio.run(run)
    assert set(tools) == {"list_tasks", "classify", "score"}
    assert all(t.annotations and t.annotations.read_only_hint for t in tools.values())
    assert not result.is_error
    data = result.structured_content
    assert data["calibrated"] is True and data["top"] == "spam"
    assert data["model"] == MODEL


def test_mcp_unknown_task_is_a_tool_error():
    async def run():
        async with Client(build_mcp(service())) as c:
            return await c.call_tool("classify", {"task": "nope", "inputs": {}})

    assert anyio.run(run).is_error


def test_classify_logs_one_line_with_the_request_id(client, calls):
    r = client.post(
        "/v1/classify",
        json={"task": "spam", "inputs": {"message": ""}, "caller": "kmon"},
    )
    body = r.json()
    assert len(body["request_id"]) == 26
    [line] = logged(calls)
    assert line["request_id"] == body["request_id"]
    assert line["caller"] == "kmon"
    assert line["task"] == "spam" and line["inputs"] == {"message": ""}
    assert line["probs"] == body["probs"] and line["top"] == body["top"]
    assert line["calibrated"] is True and line["model"] == MODEL
    assert line["prompt_sha256"] == client.get("/v1/tasks").json()[0]["prompt_sha256"]
    assert {"ts", "latency_ms", "raw"} <= set(line)
    assert stat.S_IMODE(calls.path().stat().st_mode) == 0o600
    assert stat.S_IMODE(calls.directory.stat().st_mode) == 0o700


def test_request_ids_are_distinct_and_caller_is_optional(client, calls):
    ids = {
        client.post(
            "/v1/classify", json={"task": "raw", "inputs": {"message": ""}}
        ).json()["request_id"]
        for _ in range(3)
    }
    assert len(ids) == 3
    lines = logged(calls)
    assert len(lines) == 3 and all(ln["caller"] is None for ln in lines)


def test_caller_header_is_used_when_the_body_has_none(client, calls):
    client.post(
        "/v1/classify",
        json={"task": "spam", "inputs": {"message": ""}},
        headers={"X-Homelab-Agent": "claude-kubs0"},
    )
    client.post(
        "/v1/classify",
        json={"task": "spam", "inputs": {"message": ""}, "caller": "kmon"},
        headers={"X-Homelab-Agent": "claude-kubs0"},
    )
    assert [ln["caller"] for ln in logged(calls)] == ["claude-kubs0", "kmon"]


@pytest.mark.parametrize("caller", [3, "", "x" * 65])
def test_bad_caller_is_422(client, calls, caller):
    r = client.post(
        "/v1/classify",
        json={"task": "spam", "inputs": {"message": ""}, "caller": caller},
    )
    assert r.status_code == 422
    assert logged(calls) == []


def test_errors_and_score_are_not_logged(client, calls):
    client.post("/v1/classify", json={"task": "nope", "inputs": {}})
    client.post("/v1/score", json={"prompt": ":", "choices": ["spam", "ham"]})
    assert logged(calls) == []


def test_a_failing_log_never_fails_the_call(tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("")
    svc = service(calls=CallLog(blocker / "calls"))  # parent is a file
    assert svc.calls and not svc.calls.status()["writable"]
    with TestClient(create_app(svc)) as c:
        for _ in range(2):
            r = c.post("/v1/classify", json={"task": "spam", "inputs": {"message": ""}})
            assert r.status_code == 200 and r.json()["request_id"]
        assert c.get("/healthz").json()["call_log"]["writable"] is False
    # Warned once, not once per call.
    assert capsys.readouterr().err.count("call log write") == 1


def test_healthz_reports_the_call_log(client, calls):
    log = client.get("/healthz").json()["call_log"]
    assert log == {"path": str(calls.path()), "writable": True}


def test_mcp_classify_carries_the_request_id_and_caller(calls):
    async def run():
        async with Client(build_mcp(service(calls=calls))) as c:
            return await c.call_tool(
                "classify",
                {"task": "spam", "inputs": {"message": ""}, "caller": "kmon"},
            )

    data = anyio.run(run).structured_content
    [line] = logged(calls)
    assert data["request_id"] == line["request_id"]
    assert line["caller"] == "kmon"


def test_serve_bounds_graceful_shutdown(monkeypatch):
    # tailscale serve holds keep-alive connections open; an unbounded
    # graceful shutdown waits on them until systemd's 90 s SIGKILL (#3248).
    import uvicorn

    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw))
    serve(object(), "127.0.0.1", 7780)
    assert seen == {
        "host": "127.0.0.1",
        "port": 7780,
        "timeout_graceful_shutdown": GRACEFUL_SHUTDOWN_S,
    }
    assert GRACEFUL_SHUTDOWN_S <= 10


def test_unit_stop_timeout_backstops_the_graceful_shutdown():
    unit = (ROOT / "deploy" / "kodds.service").read_text().splitlines()
    [line] = [x for x in unit if x.startswith("TimeoutStopSec=")]
    stop = int(line.split("=", 1)[1])
    # Long enough that uvicorn's own timeout fires first, short of the 90 s default.
    assert GRACEFUL_SHUTDOWN_S < stop < 90
