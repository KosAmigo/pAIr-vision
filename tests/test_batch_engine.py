"""묶음 추론 엔진(BatchedDetector) 시험: 가짜 백엔드로 단계별 묶음·빠른 프레임 먼저 끝남·실패 격리를 본다."""

import threading
import time

import pytest

from pair_vision.models.base import PHONE, WEBCAM, Frame
from pair_vision.models.batch import BatchedDetector, FakeBatchBackend, pad_bucket


def marker(state: str) -> bytes:
    return b"\xff\xd8\xff\xe0FAKE:" + state.encode()


def frame(key: str, channel: str = PHONE, seq: int = 1, state: str = "NORMAL") -> Frame:
    return Frame(key, channel, seq, 1000 + seq, marker(state))


@pytest.fixture
def batches():
    return []


def make(batches, **kw):
    det = BatchedDetector(FakeBatchBackend("normal", **kw.pop("backend", {})), **kw)
    det.batch_observer = lambda stage, n: batches.append((stage, n))
    det.load()
    return det


def test_pad_bucket_is_next_power_of_two_with_cap():
    assert [pad_bucket(n, 32) for n in (1, 2, 3, 5, 9, 17, 40)] == [1, 2, 4, 8, 16, 32, 32]


def test_every_frame_gets_its_own_detection(batches):
    det = make(batches)
    try:
        frames = [frame(f"k{i}") for i in range(5)] + [frame("k0", WEBCAM, state="NO_FACE")]
        out = det.run(frames)
        assert [(d.key, d.channel) for d in out] == [(f.key, f.channel) for f in frames]  # 같은 순서
        assert out[-1].webcam is not None and out[-1].webcam.face_count == 0
        assert all(len(d.hands) == 2 for d in out[:5])
    finally:
        det.close()


def test_frames_waiting_together_are_run_as_one_batch(batches):
    # 첫 묶음이 도는 동안(50ms) 들어온 프레임들은 다음 묶음 하나로 모인다
    det = make(batches, max_batch=16, wait_ms=0, backend={"stage_ms": {"hands": 50.0}})
    try:
        futs = [det.submit(frame("first"))]
        time.sleep(0.01)
        futs += [det.submit(frame(f"k{i}")) for i in range(10)]
        for f in futs:
            f.result(timeout=5)
        sizes = [n for stage, n in batches if stage == "hands"]
        assert sizes[0] == 1 and max(sizes) == 10
    finally:
        det.close()


def test_max_batch_limits_batch_size(batches):
    det = make(batches, max_batch=4, wait_ms=30)
    try:
        det.run([frame(f"k{i}") for i in range(10)])
        assert max(n for _, n in batches) <= 4
    finally:
        det.close()


def test_fast_webcam_frame_does_not_wait_for_slow_phone_batch(batches):
    """'묶음이 다 끝나야 다음 묶음' 구조가 아니다: 느린 휴대폰 묶음이 도는 중에 온 웹캠 프레임은
    그 묶음이 끝나자마자 처리되고, 뒤에 온 휴대폰 묶음(300ms)이나 앞 묶음의 나머지 단계를 기다리지 않는다."""
    det = make(batches, wait_ms=0, backend={"stage_ms": {"hands": 300.0, "webcam": 5.0}})
    try:
        slow = det.submit(frame("slow"))
        time.sleep(0.05)
        t0 = time.monotonic()
        web = det.submit(frame("w", WEBCAM))
        time.sleep(0.02)
        later = [det.submit(frame(f"p{i}")) for i in range(3)]  # 웹캠보다 늦게 온 휴대폰 프레임
        web.result(timeout=5)
        web_done = time.monotonic() - t0
        slow.result(timeout=5)
        for f in later:
            f.result(timeout=5)
        later_done = time.monotonic() - t0
        assert web_done < 0.4 and later_done > 0.45
    finally:
        det.close()


def test_failing_frame_is_isolated_from_its_batch(batches):
    det = make(batches, wait_ms=20, backend={"fail_keys": {"bad"}})
    try:
        futs = {k: det.submit(frame(k)) for k in ("a", "bad", "b")}
        assert futs["a"].result(timeout=5).key == "a"
        assert futs["b"].result(timeout=5).key == "b"
        with pytest.raises(RuntimeError):
            futs["bad"].result(timeout=5)
        out = det.run([frame("c"), frame("bad")])  # 동기 실행은 실패한 프레임만 건너뛴다
        assert [d.key for d in out] == ["c"]
    finally:
        det.close()


def test_stats_report_per_frame_stage_time_and_batches(batches):
    det = make(batches, backend={"stage_ms": {"hands": 5.0}})
    try:
        det.run([frame(f"k{i}") for i in range(3)])
        stats = det.stats()
        calls, seconds = stats["hands"]
        assert calls == 3 and seconds > 0  # 호출 수 = 처리한 프레임 수(묶음 시간을 프레임 수로 나눠 평균 ms 를 본다)
        assert "prepare" in stats
    finally:
        det.close()


def test_close_fails_pending_work_and_is_idempotent(batches):
    det = make(batches, backend={"stage_ms": {"hands": 200.0}})
    futs = [det.submit(frame(f"k{i}")) for i in range(3)]
    det.close()
    det.close()
    for f in futs:  # 전처리 풀·대기열에 있던 작업도 오류로 끝나야 한다(영원히 기다리지 않음)
        try:
            f.result(timeout=2)
        except RuntimeError:
            pass
        assert f.done()
    assert det.submit(frame("late")).exception(timeout=1) is not None  # 종료 뒤 제출은 바로 오류


def test_engine_runs_stages_on_single_thread(batches):
    seen = set()

    class Probe(FakeBatchBackend):
        def run_stage(self, stage, jobs):
            seen.add(threading.get_ident())
            super().run_stage(stage, jobs)

    det = BatchedDetector(Probe("normal"))
    det.load()
    try:
        det.run([frame(f"k{i}") for i in range(6)] + [frame("w", WEBCAM)])
        assert len(seen) == 1  # GPU 단계는 엔진 스레드 하나에서만(모델 1개를 응시자들이 나눠 씀)
    finally:
        det.close()
