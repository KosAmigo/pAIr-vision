"""WebSocket /v1/stream -> 가짜 모델 -> 규칙 엔진 -> 서명된 이벤트 콜백(HANDS_OUT) 전체 흐름."""

import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from conftest import SECRET, new_key
from pair_vision.api.app import create_app, build_core
from pair_vision.auth.signing import verify
from pair_vision.evidence.callback import CallbackSender
from pair_vision.evidence.events import make_client_id
from pair_vision.session.frames import encode_frame

T0 = 1764000000000


def jpeg(state: str) -> bytes:
    return b"\xff\xd8\xff\xe0FAKE:" + state.encode()


class Recorder:
    """콜백 요청을 모으는 MockTransport 핸들러."""

    def __init__(self):
        self.requests: list[httpx.Request] = []

    def __call__(self, request):
        self.requests.append(request)
        return httpx.Response(201, json={"id": 1, "duplicate": False})


def wait_for(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def env(settings):
    rec = Recorder()
    client = httpx.AsyncClient(transport=httpx.MockTransport(rec))
    sender = CallbackSender(settings.callback_url, "k1", SECRET, client)
    with TestClient(create_app(settings, sink=sender)) as tc:
        yield tc, rec, sender


def send_and_wait(ws, core, key, seq, recv_ms, state, channel="phone"):
    before = core.pipeline.processed
    ws.send_bytes(encode_frame(key, channel, seq, recv_ms, jpeg(state)))
    assert wait_for(lambda: core.pipeline.processed > before)


def test_hands_out_event_is_signed_and_sent(env, auth):
    tc, rec, sender = env
    key = new_key()
    tc.post(f"/v1/sessions/{key}/start", json={"mode": "exam", "startedAtMs": T0}, headers=auth)
    core = tc.app.state.core
    with tc.websocket_connect("/v1/stream", headers=auth) as ws:
        for i in range(9):  # 0..8초, 두 손 모두 밖
            send_and_wait(ws, core, key, i + 1, T0 + i * 1000, "BOTH_OUT")
    assert wait_for(lambda: len(rec.requests) >= 1)
    req = rec.requests[0]
    body = json.loads(req.content)
    assert body["type"] == "HANDS_OUT" and body["source"] == "PHONE_CAM"
    assert body["sessionKey"] == key and body["final"] is False
    assert body["startedAtMs"] == T0 and body["durationMs"] == 7000
    assert body["rulesVersion"] == "2026-11-26.2"
    assert body["clientId"] == make_client_id(key, "HANDS_OUT", T0)
    assert req.headers["x-vision-key-id"] == "k1"
    ts = req.headers["x-vision-timestamp"]
    assert verify(SECRET, ts, req.content, req.headers["x-vision-signature"])

    # 세션 종료 -> 진행 중인 이상 상태가 final 로 마무리되어 같은 clientId 로 나간다
    r = tc.post(f"/v1/sessions/{key}/end", headers=auth)
    assert r.json() == {"flushedEvents": 1}
    assert wait_for(lambda: len(rec.requests) >= 2)
    final = json.loads(rec.requests[1].content)
    assert final["final"] is True and final["clientId"] == body["clientId"]
    assert final["durationMs"] == 8000


def test_unknown_session_error_throttled_to_once_per_10s(env, auth):
    tc, _, _ = env
    key = new_key()
    with tc.websocket_connect("/v1/stream", headers=auth) as ws:
        for i in range(5):
            ws.send_bytes(encode_frame(key, "phone", i, T0 + i, jpeg("NORMAL")))
        msg = ws.receive_json()
        assert msg == {"type": "error", "code": "UNKNOWN_SESSION", "sessionKey": key}
        # 추가 오류는 없어야 한다: 다른 키로 확인 메시지가 먼저 오는지 본다
        other = new_key()
        ws.send_bytes(encode_frame(other, "phone", 1, T0, jpeg("NORMAL")))
        assert ws.receive_json()["sessionKey"] == other


def test_stream_rejects_bad_bearer(env):
    tc, _, _ = env
    with pytest.raises(WebSocketDisconnect):
        with tc.websocket_connect("/v1/stream", headers={"Authorization": "Bearer nope"}):
            pass
    with pytest.raises(WebSocketDisconnect):
        with tc.websocket_connect("/v1/stream"):
            pass


def test_state_message_for_exam_sessions(env, auth):
    tc, _, _ = env
    key = new_key()
    tc.post(f"/v1/sessions/{key}/start", json={"mode": "exam", "startedAtMs": T0}, headers=auth)
    core = tc.app.state.core
    with tc.websocket_connect("/v1/stream", headers=auth) as ws:
        send_and_wait(ws, core, key, 1, T0, "NORMAL")
        msg = ws.receive_json()  # state_interval_s=0.2
        assert msg["type"] == "state" and msg["sessionKey"] == key
        assert msg["calibration"] == "pending" and msg["phone"] == "ok" and msg["webcam"] == "idle"
        assert msg["lastAnalyzedMs"] == T0 and set(msg["analysisFps"]) == {"webcam", "phone"}


def test_check_mode_returns_check_without_events(env, auth):
    tc, rec, _ = env
    key = new_key()
    tc.post(f"/v1/sessions/{key}/start", json={"mode": "check"}, headers=auth)
    core = tc.app.state.core
    with tc.websocket_connect("/v1/stream", headers=auth) as ws:
        for i in range(5):
            send_and_wait(ws, core, key, i + 1, T0 + i * 1000, "NORMAL")
        msgs = []
        while not msgs or msgs[-1]["holdMs"] < 3000:
            m = ws.receive_json()
            if m["type"] == "check":
                msgs.append(m)
    last = msgs[-1]
    assert last["passed"] is True and all(last["items"].values()) and last["hints"] == []
    assert rec.requests == []  # check 모드는 이벤트 없음


def test_one_hand_missing_does_not_pass_check(env, auth):
    tc, _, _ = env
    key = new_key()
    tc.post(f"/v1/sessions/{key}/start", json={"mode": "check"}, headers=auth)
    core = tc.app.state.core
    with tc.websocket_connect("/v1/stream", headers=auth) as ws:
        send_and_wait(ws, core, key, 1, T0, "ONE_HAND")
        m = ws.receive_json()
        while m["type"] != "check":
            m = ws.receive_json()
    assert m["passed"] is False and m["items"]["rightHand"] is False
    assert "RIGHT_HAND_MISSING" in m["hints"]


def test_webcam_calibration_completes_after_three_seconds(env, auth):
    tc, _, _ = env
    key = new_key()
    tc.post(f"/v1/sessions/{key}/start", json={"mode": "exam", "startedAtMs": T0}, headers=auth)
    core = tc.app.state.core
    with tc.websocket_connect("/v1/stream", headers=auth) as ws:
        for i in range(4):
            send_and_wait(ws, core, key, i + 1, T0 + i * 1000, "NORMAL", channel="webcam")
    assert core.registry.get(key).calibration == "done"


def test_build_core_without_callback_url_uses_null_sink(settings):
    from dataclasses import replace

    from pair_vision.evidence.callback import NullSender

    core = build_core(replace(settings, callback_url="", callback_secret=""))
    assert isinstance(core.sink, NullSender)
