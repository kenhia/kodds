import threading
import time

import anyio
import pytest
from mcp import Client
from starlette.testclient import TestClient
from test_scorer import plain
from test_tasks import spam_model, spam_task

from kodds.scorer import Scorer
from kodds.service import (
    BadRequest,
    Service,
    UnknownTask,
    build_mcp,
    create_app,
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
def client():
    with TestClient(create_app(service())) as c:
        yield c


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
