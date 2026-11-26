"""프로세스 풀 Detector(워커 병렬 처리) 시험: 가짜 모델로 워커 프로세스를 실제로 띄운다."""

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from conftest import SECRET, new_key
from pair_vision.api.app import create_app
from pair_vision.config import Settings
from pair_vision.models.base import PHONE, WEBCAM, Frame
from pair_vision.models.fake import FakeDetector as FakeDetectorBase
from pair_vision.models.pool import PooledDetector
from pair_vision.evidence.callback import NullSender
from pair_vision.metrics import Metrics
from pair_vision.rules.config import parse_rules
from pair_vision.session.frames import encode_frame

SPEC = {"kind": "fake", "scenario": "normal"}


def marker(state: str) -> bytes:
    return b"\xff\xd8\xff\xe0FAKE:" + state.encode()


@pytest.fixture(scope="module")
def pool():
    det = PooledDetector(SPEC, workers=2)
    det.load()
    yield det
    det.close()


def test_pool_returns_detections_for_every_frame_across_workers(pool):
    frames = [Frame(f"key{i}", PHONE, 1, 1000, marker("BOTH_OUT")) for i in range(5)]
    frames.append(Frame("key0", WEBCAM, 1, 1000, marker("NO_FACE")))
    out = pool.run(frames)
    assert sorted((d.key, d.channel) for d in out) == sorted((f.key, f.channel) for f in frames)
    phone = [d for d in out if d.channel == PHONE]
    assert all(len(d.hands) == 2 and d.keyboard is not None for d in phone)  # 가짜 BOTH_OUT: 손 둘
    web = [d for d in out if d.channel == WEBCAM][0]
    assert web.webcam is not None and web.webcam.face_count == 0


def test_same_key_always_goes_to_same_worker_and_load_is_balanced(pool):
    keys = [f"k{i}" for i in range(8)]
    first = {k: pool.worker_for(k) for k in keys}
    assert first == {k: pool.worker_for(k) for k in keys}  # 고정 배정
    counts = [list(first.values()).count(i) for i in range(2)]
    assert counts == [4, 4]  # 균등 분배


def test_forget_releases_slot(pool):
    key = "forget-me"
    idx = pool.worker_for(key)
    before = pool._count[idx]
    pool.forget(key)
    assert pool._count[idx] == before - 1
    assert key not in pool._assign


def test_run_async_runs_workers_concurrently(pool):
    frames = [Frame(f"conc{i}", PHONE, 1, 1000, marker("NORMAL")) for i in range(6)]
    out = asyncio.run(pool.run_async(frames))
    assert len(out) == 6


def test_pool_needs_two_workers():
    with pytest.raises(ValueError):
        PooledDetector(SPEC, workers=1)


def test_settings_and_config_for_workers():
    env = {"VISION_API_KEY": "k", "VISION_WORKERS": "4", "VISION_WORKER_THREADS": "2"}
    s = Settings.from_env(env)
    assert (s.workers, s.worker_threads) == (4, 2)
    assert Settings.from_env({"VISION_API_KEY": "k"}).workers == 1
    assert parse_rules({"version": "v", "profiles": {"default": {}}, "models": {"pose_every": 5}}).models.pose_every == 5
    with pytest.raises(ValueError, match="pose_every"):
        parse_rules({"version": "v", "profiles": {"default": {}}, "models": {"pose_every": 0}})


def wait_for(cond, timeout=20.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_end_to_end_with_two_workers(settings, auth):
    """워커 2개로 띄운 서버에서도 스트림 -> 규칙 -> 이벤트가 그대로 동작한다."""
    cfg = Settings(**{**settings.__dict__, "workers": 2, "max_sessions": 4})
    sent = []

    class Sink:
        def enqueue(self, body):
            sent.append(body)

        async def run(self):
            pass

    with TestClient(create_app(cfg, sink=Sink())) as tc:
        assert tc.get("/readyz", headers=auth).json()["modelsLoaded"] is True
        key = new_key()
        assert tc.post(f"/v1/sessions/{key}/start", json={"mode": "exam", "startedAtMs": 0}, headers=auth).status_code == 200
        t0 = 1_764_000_000_000
        with tc.websocket_connect("/v1/stream", headers=auth) as ws:
            core = tc.app.state.core
            for i in range(9):  # 두 손 모두 밖 8초 이상 -> HANDS_OUT
                before = core.pipeline.processed
                ws.send_bytes(encode_frame(key, "phone", i + 1, t0 + i * 1000, marker("BOTH_OUT")))
                assert wait_for(lambda: core.pipeline.processed > before)
        assert wait_for(lambda: any(e["type"] == "HANDS_OUT" for e in sent))
    assert SECRET  # conftest 의 비밀은 서명 시험용이라 여기서는 쓰지 않는다


def test_cadence_runs_detector_only_at_interval():
    from pair_vision.models.postprocess import Cadence

    c = Cadence(3000)
    assert c.due("k", 0)
    c.mark("k", 0)
    assert not c.due("k", 2999) and c.due("k", 3000)
    assert c.due("other", 10)  # 세션마다 따로
    c.forget("k")
    assert c.due("k", 100)
    assert Cadence(0).due("k", 5)  # 0 이면 항상


def test_stage_timer_snapshot_and_metrics_endpoint(settings, auth):
    from pair_vision.models.postprocess import StageTimer

    t = StageTimer()
    t.add("dfine", 0.1)
    t.add("dfine", 0.3)
    assert t.snapshot() == {"dfine": (2, pytest.approx(0.4))}

    class Timed(FakeDetectorBase):
        def stats(self):
            return {"dfine": (4.0, 0.4)}

    with TestClient(create_app(settings, detector=Timed(), sink=NullSender(Metrics()))) as tc:
        text = tc.get("/metrics", headers=auth).text
    assert 'pair_vision_stage_calls_total{stage="dfine"} 4.0' in text
    assert 'pair_vision_stage_seconds_total{stage="dfine"} 0.4' in text


def test_note_gap_must_cover_dfine_interval():
    base = {"version": "v", "profiles": {"default": {}}}
    with pytest.raises(ValueError, match="note.gap_ms"):
        parse_rules({**base, "models": {"dfine_interval_ms": 10000}})  # 기본 노트 gap 7000 < 2*10000
    assert parse_rules(base).models.dfine_interval_ms == 3000
