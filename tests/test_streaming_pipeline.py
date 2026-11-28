"""연속 처리 파이프라인(묶음 추론 엔진) 시험: 칸마다 처리 중 1장, 끝나면 바로 다음 프레임, 지연·간격 지표."""

import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from conftest import SECRET, new_key
from pair_vision.api.app import create_app
from pair_vision.config import Settings
from pair_vision.evidence.callback import CallbackSender
from pair_vision.models.base import PHONE
from pair_vision.models.batch import BatchedDetector, FakeBatchBackend
from pair_vision.models.fake import create_detector
from pair_vision.session.frames import ParsedFrame, encode_frame
from pair_vision.session.mailbox import Mailboxes

T0 = 1764000000000


def jpeg(state: str) -> bytes:
    return b"\xff\xd8\xff\xe0FAKE:" + state.encode()


def wait_for(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def metric(text: str, name: str) -> float:
    total = 0.0
    for line in text.splitlines():
        if line.startswith(name) and not line.startswith("#"):
            total += float(line.rsplit(" ", 1)[1])
    return total


def test_take_ready_skips_busy_slots_and_keeps_their_frame():
    mb = Mailboxes()
    mb.put(ParsedFrame("a", PHONE, 1, 0, b""))
    mb.put(ParsedFrame("b", PHONE, 1, 0, b""))
    got = mb.take_ready({("a", PHONE)})
    assert [f.key for f in got] == ["b"]
    assert [f.key for f in mb.take_ready(set())] == ["a"]  # 바쁜 칸의 프레임은 남아 있다


class Recorder:
    def __init__(self):
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return httpx.Response(201, json={"id": 1, "duplicate": False})


@pytest.fixture
def make_env(settings):
    clients = []

    def _make(stage_ms=None):
        rec = Recorder()
        sender = CallbackSender(settings.callback_url, "k1", SECRET, httpx.AsyncClient(transport=httpx.MockTransport(rec)))
        det = BatchedDetector(FakeBatchBackend("normal", stage_ms=stage_ms or {}), wait_ms=0)
        tc = TestClient(create_app(settings, detector=det, sink=sender))
        tc.__enter__()
        clients.append(tc)
        return tc, rec

    yield _make
    for tc in clients:
        tc.__exit__(None, None, None)


def test_streaming_pipeline_makes_signed_hands_out_event(make_env, auth):
    tc, rec = make_env()
    core = tc.app.state.core
    key = new_key()
    tc.post(f"/v1/sessions/{key}/start", json={"mode": "exam", "startedAtMs": T0}, headers=auth)
    with tc.websocket_connect("/v1/stream", headers=auth) as ws:
        for i in range(9):
            before = core.pipeline.processed
            ws.send_bytes(encode_frame(key, "phone", i + 1, T0 + i * 1000, jpeg("BOTH_OUT")))
            assert wait_for(lambda: core.pipeline.processed > before)
    assert wait_for(lambda: len(rec.requests) >= 1)
    body = json.loads(rec.requests[0].content)
    assert (body["type"], body["durationMs"], body["rulesVersion"]) == ("HANDS_OUT", 7000, "2026-11-27.1")


def test_one_frame_in_flight_per_slot_and_newest_waiting_frame_wins(make_env, auth):
    tc, _ = make_env({"hands": 300.0})
    core = tc.app.state.core
    key = new_key()
    tc.post(f"/v1/sessions/{key}/start", json={"mode": "exam", "startedAtMs": T0}, headers=auth)
    with tc.websocket_connect("/v1/stream", headers=auth) as ws:
        for seq in (1, 2, 3):  # 1 은 처리 중, 2 는 대기, 3 이 2 를 밀어낸다
            ws.send_bytes(encode_frame(key, "phone", seq, T0 + seq * 1000, jpeg("NORMAL")))
            time.sleep(0.03)
        assert wait_for(lambda: core.pipeline.processed == 2, timeout=3)
        time.sleep(0.4)
    text = tc.get("/metrics", headers=auth).text
    assert core.pipeline.processed == 2
    assert metric(text, 'pair_vision_frames_dropped_total{reason="replaced"}') == 1
    # 처리된 두 프레임의 recvMs 간격 2초(1 -> 3)가 간격 지표에 들어간다
    assert metric(text, 'pair_vision_processed_gap_seconds_count{channel="phone"}') == 1
    assert metric(text, 'pair_vision_processed_gap_seconds_bucket{channel="phone",le="2.0"}') == 1
    assert metric(text, 'pair_vision_processed_gap_seconds_bucket{channel="phone",le="1.5"}') == 0


def test_other_slots_are_not_blocked_by_a_slow_frame(make_env, auth):
    tc, _ = make_env({"hands": 400.0, "webcam": 1.0})
    core = tc.app.state.core
    key = new_key()
    tc.post(f"/v1/sessions/{key}/start", json={"mode": "exam", "startedAtMs": T0}, headers=auth)
    with tc.websocket_connect("/v1/stream", headers=auth) as ws:
        ws.send_bytes(encode_frame(key, "phone", 1, T0, jpeg("NORMAL")))
        time.sleep(0.05)
        t0 = time.time()
        for seq in range(1, 4):  # 웹캠 프레임은 휴대폰 묶음이 도는 동안 기다리지 않고 차례로 처리된다(엔진이 휴대폰 단계 뒤로 줄 세움)
            ws.send_bytes(encode_frame(key, "webcam", seq, T0 + seq * 100, jpeg("FRONT")))
            assert wait_for(lambda s=seq: core.pipeline.processed >= s + 1, timeout=3)
        assert time.time() - t0 < 1.0
    text = tc.get("/metrics", headers=auth).text
    assert metric(text, "pair_vision_frame_latency_seconds_count") >= 3
    assert metric(text, 'pair_vision_batch_size_count{stage="webcam"}') >= 1
    assert metric(text, 'pair_vision_stage_calls_total{stage="prepare"}') >= 3


def test_settings_batch_env_and_detector_selection(rules):
    s = Settings.from_env({"VISION_API_KEY": "k", "VISION_BATCH": "1", "VISION_BATCH_MAX": "8",
                           "VISION_BATCH_WAIT_MS": "5", "VISION_PREP_THREADS": "2", "VISION_OWL_MAX_BATCH": "2"})
    assert (s.batch, s.batch_max, s.batch_wait_ms, s.prep_threads, s.owl_max_batch) == (True, 8, 5.0, 2, 2)
    assert Settings.from_env({"VISION_API_KEY": "k"}).batch is False
    assert Settings.from_env({"VISION_API_KEY": "k"}).device_preprocess is None
    assert Settings.from_env({"VISION_API_KEY": "k", "VISION_DEVICE_PREPROCESS": "1"}).device_preprocess is True
    assert Settings.from_env({"VISION_API_KEY": "k", "VISION_FP16_SKIP": "dfine, owlv2"}).fp16_skip == {"dfine", "owlv2"}
    from pair_vision.config import ConfigError

    with pytest.raises(ConfigError):
        Settings.from_env({"VISION_API_KEY": "k", "VISION_FP16_SKIP": "rtmw"})
    det = create_detector("fake", "normal", batch=True)
    assert isinstance(det, BatchedDetector) and det.streaming
    det2 = create_detector("fake", "normal", batch=True, workers=4)  # 묶음 추론은 모델 1프로세스: 워커 수는 무시
    assert isinstance(det2, BatchedDetector)
