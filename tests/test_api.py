import pytest
from fastapi.testclient import TestClient

from conftest import API_KEY, NEXT_KEY, new_key
from pair_vision.api.app import create_app
from pair_vision.evidence.callback import NullSender


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, sink=NullSender())) as c:
        yield c


def start(client, auth, key, **body):
    payload = {"mode": "exam", "startedAtMs": 1764000000000, **body}
    return client.post(f"/v1/sessions/{key}/start", json=payload, headers=auth)


def test_auth_required_and_both_keys_work(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 401
    bad = client.get("/readyz", headers={"Authorization": "Bearer bad"})
    assert bad.json()["code"] == "UNAUTHORIZED"
    for k in (API_KEY, NEXT_KEY):
        assert client.get("/readyz", headers={"Authorization": f"Bearer {k}"}).status_code == 200
    assert client.post(f"/v1/sessions/{new_key()}/end").status_code == 401
    assert client.post("/v1/admin/drain").status_code == 401
    assert client.get("/metrics").status_code == 401


def test_readyz_body(client, auth):
    body = client.get("/readyz", headers=auth).json()
    assert body["ready"] and body["modelsLoaded"] and body["gpu"] is False
    assert body["maxSessions"] == 2 and body["draining"] is False
    assert body["rulesVersion"] == "2026-11-26.2"


def test_start_is_idempotent(client, auth):
    key = new_key()
    first = start(client, auth, key).json()
    assert first == {"accepted": True, "resumed": False, "sessions": 1, "maxSessions": 2}
    again = start(client, auth, key).json()
    assert again["resumed"] is True and again["sessions"] == 1


def test_recalibrate_resets_calibration(client, auth):
    key = new_key()
    start(client, auth, key)
    session = client.app.state.core.registry.get(key)
    session.calibration = "done"
    assert start(client, auth, key, recalibrate=True).json()["resumed"] is True
    assert session.calibration == "pending"


def test_over_capacity_503_with_retry_after(client, auth):
    start(client, auth, new_key())
    start(client, auth, new_key())
    r = start(client, auth, new_key())
    assert r.status_code == 503 and r.json()["code"] == "OVER_CAPACITY"
    assert r.headers["retry-after"] == "5"


def test_existing_key_still_resumes_when_full(client, auth):
    k = new_key()
    start(client, auth, k)
    start(client, auth, new_key())
    assert start(client, auth, k).status_code == 200


def test_drain_rejects_new_but_keeps_existing(client, auth):
    k = new_key()
    start(client, auth, k)
    assert client.post("/v1/admin/drain", headers=auth).status_code == 202
    r = start(client, auth, new_key())
    assert r.status_code == 503 and r.json()["code"] == "DRAINING"
    assert start(client, auth, k).json()["resumed"] is True
    assert client.get("/readyz", headers=auth).json()["draining"] is True


@pytest.mark.parametrize("body", [{"mode": "weird"}, {}, {"mode": "exam", "rulesProfile": "nope"}])
def test_bad_start_requests_400(client, auth, body):
    r = client.post(f"/v1/sessions/{new_key()}/start", json=body, headers=auth)
    assert r.status_code == 400 and r.json()["code"] == "BAD_REQUEST"


def test_bad_key_format_400(client, auth):
    assert start(client, auth, "too-short").status_code == 400


def test_mode_switch_and_unknown_key_404(client, auth):
    r = client.post(f"/v1/sessions/{new_key()}/mode", json={"mode": "exam"}, headers=auth)
    assert r.status_code == 404 and r.json()["code"] == "UNKNOWN_SESSION"
    k = new_key()
    start(client, auth, k, mode="check")
    body = {"mode": "exam", "startedAtMs": 5}
    assert client.post(f"/v1/sessions/{k}/mode", json=body, headers=auth).status_code == 200
    assert client.app.state.core.registry.get(k).mode == "exam"


def test_end_unknown_key_is_200_and_frees_slot(client, auth):
    assert client.post(f"/v1/sessions/{new_key()}/end", headers=auth).json() == {"flushedEvents": 0}
    k = new_key()
    start(client, auth, k)
    client.post(f"/v1/sessions/{k}/end", headers=auth)
    assert client.app.state.core.registry.get(k) is None


def test_metrics_exposes_prometheus_text(client, auth):
    text = client.get("/metrics", headers=auth).text
    assert "pair_vision_sessions" in text and "pair_vision_callback_queue_size" in text
