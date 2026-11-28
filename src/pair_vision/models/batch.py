"""묶음 추론 엔진: 모델을 한 프로세스에 두고 여러 응시자의 프레임을 단계(모델)마다 묶어 돌린다 (VISION_BATCH=1).

왜: 워커 프로세스 N개가 GPU 를 시간 분할하면 응시자 1명당 GPU 시간이 그대로 쌓여 GPU 1장 약 8명에서 막힌다(용량 시뮬레이션).
같은 모델을 여러 장 한 번에 돌리면(배치) 장당 비용이 줄어든다.

구조
- 프레임 하나 = 작업(Job). 백엔드의 `prepare`(CPU 스레드 풀: JPEG 풀기·주기 판단·CPU 전처리)가 거칠 단계 목록을 정한다
  (예: 휴대폰 = dfine? -> rtmw -> owlv2?, 웹캠 = person? -> rtmw).
- 단계마다 대기열이 있고, **엔진 스레드 하나**가 "가장 오래 기다린 작업이 있는 단계"를 골라 그 단계의 대기 작업을
  최대 `max_batch` 장까지 한 번에 돌린다. 끝난 작업은 다음 단계 대기열로 가거나 결과(Detection)를 돌려준다.
- "묶음이 다 끝나야 다음 묶음"이 아니다: 작업마다 따로 끝나므로 빠른 웹캠 프레임이 느린 OWLv2 묶음을 기다리지 않고,
  파이프라인은 끝난 칸에 곧바로 다음 프레임을 넣는다(pipeline/worker.py 의 연속 처리).
- GPU 가 놀고 있을 때 첫 작업이 오면 `wait_ms` 만큼만 더 모은다(묶음을 키우려고). 바쁠 때는 기다리지 않는다
  (바쁜 동안 쌓인 작업이 자연히 다음 묶음이 된다).
- 묶음 실행이 실패하면 그 묶음을 한 장씩 다시 돌려 실패한 프레임만 오류로 끝낸다.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, InvalidStateError, ThreadPoolExecutor
from dataclasses import dataclass, field

from pair_vision.models.base import PHONE, Detection, Detector, Frame
from pair_vision.models.postprocess import StageTimer

log = logging.getLogger("pair_vision.batch")

DEFAULT_MAX_BATCH = 16
DEFAULT_WAIT_MS = 15.0
DEFAULT_PREP_THREADS = 4


def pad_bucket(n: int, cap: int) -> int:
    """묶음 크기를 2의 거듭제곱으로 올린다(입력 모양 가짓수를 줄여 GPU 첫 실행 지연·cuDNN 탐색을 막는다). cap 을 넘지 않는다."""
    size = 1
    while size < n:
        size *= 2
    return min(size, cap)


@dataclass(eq=False)  # 객체 동일성으로 비교·해시(진행 중 작업 집합에 넣는다)
class Job:
    frame: Frame
    future: Future
    stages: deque[str] = field(default_factory=deque)
    state: dict = field(default_factory=dict)  # 단계 사이에 넘기는 값(이미지·전처리 텐서·모델 결과)
    enqueued: float = 0.0  # 지금 단계 대기열에 들어간 시각(monotonic)


class BatchBackend:
    """엔진이 부르는 모델 쪽 구현. 메서드는 엔진 스레드(run_stage·finish) 또는 CPU 풀(prepare)에서 불린다."""

    gpu: bool = False

    def load(self) -> None:
        """모델 적재."""

    def prepare(self, job: Job) -> list[str]:
        """CPU 풀에서: 디코드·주기 판단·CPU 전처리. 거칠 단계 이름 목록을 돌려준다(빈 목록이면 바로 finish)."""
        raise NotImplementedError

    def run_stage(self, stage: str, jobs: list[Job]) -> None:
        """엔진 스레드에서: 한 단계를 jobs 묶음으로 실행하고 결과를 job.state 에 넣는다. 예외면 엔진이 한 장씩 다시 돈다."""
        raise NotImplementedError

    def finish(self, job: Job) -> Detection:
        raise NotImplementedError

    def max_batch(self, stage: str, default: int) -> int:
        return default

    def forget(self, key: str) -> None:
        """세션 종료: 세션별 상태(주기·상자 캐시)를 비운다."""

    def stats(self) -> dict[str, tuple[float, float]]:
        return {}

    def close(self) -> None:
        """자원 정리."""


class BatchedDetector(Detector):
    """Detector 인터페이스 + 프레임 하나씩 끝나는 `submit`/`infer`(파이프라인 연속 처리용)."""

    name = "batch"
    streaming = True

    def __init__(
        self,
        backend: BatchBackend,
        *,
        max_batch: int = DEFAULT_MAX_BATCH,
        wait_ms: float = DEFAULT_WAIT_MS,
        prep_threads: int = DEFAULT_PREP_THREADS,
    ) -> None:
        self._backend = backend
        self._max_batch = max(1, max_batch)
        self._wait_s = max(0.0, wait_ms) / 1000
        self._prep_threads = max(1, prep_threads)
        self._queues: dict[str, deque[Job]] = {}
        self._cv = threading.Condition()
        self._stopping = False
        self._thread: threading.Thread | None = None
        self._pool: ThreadPoolExecutor | None = None
        self._timer = StageTimer()
        self._live: set[Job] = set()  # 아직 안 끝난 작업(종료 때 전부 오류로 끝내려고)
        self._live_lock = threading.Lock()
        self.batch_observer: Callable[[str, int], None] | None = None  # (단계, 묶음 크기) -> 지표
        self.gpu = False

    # --- 수명 ---
    def load(self) -> None:
        self._backend.load()
        self.gpu = self._backend.gpu
        self._pool = ThreadPoolExecutor(self._prep_threads, thread_name_prefix="vision-prep")
        self._thread = threading.Thread(target=self._loop, name="vision-batch", daemon=True)
        self._thread.start()
        log.info("묶음 추론 엔진 시작 (최대 묶음 %d장, 모으기 %.0fms, 전처리 스레드 %d, gpu=%s)",
                 self._max_batch, self._wait_s * 1000, self._prep_threads, self.gpu)

    def close(self) -> None:
        with self._cv:
            if self._stopping:
                return
            self._stopping = True
            self._queues.clear()
            self._cv.notify_all()
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
        with self._live_lock:
            pending = list(self._live)  # 대기열·전처리 풀에 남은 작업 모두(풀이 취소한 작업의 Future 는 따로 끝내야 한다)
        for job in pending:
            self._fail(job, RuntimeError("서버 종료"))
        if self._thread is not None:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                log.warning("엔진 스레드가 5초 안에 끝나지 않았습니다(진행 중 묶음). 모델 정리는 건너뜁니다")
                return
        self._backend.close()

    def forget(self, key: str) -> None:
        self._backend.forget(key)

    def stats(self) -> dict[str, tuple[float, float]]:
        return {**self._backend.stats(), **self._timer.snapshot()}

    # --- 제출 ---
    def submit(self, frame: Frame) -> Future:
        """프레임 하나를 넣는다. 결과는 Future(Detection). 실패하면 Future 에 예외가 들어간다."""
        fut: Future = Future()
        job = Job(frame, fut)
        if self._stopping or self._pool is None:
            fut.set_exception(RuntimeError("엔진이 동작 중이 아님"))
            return fut
        with self._live_lock:
            self._live.add(job)
        fut.add_done_callback(lambda _f, j=job: self._forget_job(j))
        try:
            self._pool.submit(self._prepare, job)
        except RuntimeError as exc:  # 종료 중
            self._fail(job, exc)
        return fut

    def _forget_job(self, job: Job) -> None:
        with self._live_lock:
            self._live.discard(job)

    async def infer(self, frame: Frame) -> Detection:
        return await asyncio.wrap_future(self.submit(frame))

    def run(self, frames: Sequence[Frame]) -> list[Detection]:
        """동기 실행(시험·벤치용). 실패한 프레임은 건너뛰고, 전부 실패하면 예외."""
        futs = [self.submit(f) for f in frames]
        out: list[Detection] = []
        errors: list[BaseException] = []
        for fut in futs:
            try:
                out.append(fut.result())
            except Exception as exc:  # noqa: BLE001 - 프레임 하나 때문에 묶음 전체를 잃지 않는다
                errors.append(exc)
        if errors and not out:
            raise RuntimeError("모든 프레임 분석 실패") from errors[0]
        return out

    async def run_async(self, frames: Sequence[Frame]) -> list[Detection]:
        results = await asyncio.gather(*(self.infer(f) for f in frames), return_exceptions=True)
        out = [r for r in results if isinstance(r, Detection)]
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors and not out:
            raise errors[0]
        return out

    # --- 내부 ---
    def _prepare(self, job: Job) -> None:
        t = time.perf_counter()
        try:
            stages = self._backend.prepare(job)
        except Exception as exc:  # noqa: BLE001
            self._fail(job, exc)
            return
        finally:
            self._timer.add("prepare", time.perf_counter() - t)
        job.stages = deque(stages)
        try:
            with self._cv:
                self._advance(job)
        except Exception as exc:  # noqa: BLE001 - 어떤 경우에도 Future 는 끝낸다(안 그러면 그 칸이 영원히 막힌다)
            log.exception("작업 넘기기 실패")
            self._fail(job, exc)

    def _advance(self, job: Job) -> None:
        """다음 단계 대기열로 보내거나 끝낸다. self._cv 를 잡은 상태에서 부른다."""
        if job.stages:
            stage = job.stages.popleft()
            if self._stopping:
                self._fail(job, RuntimeError("서버 종료"))
                return
            job.enqueued = time.monotonic()
            self._queues.setdefault(stage, deque()).append(job)
            self._cv.notify()
            return
        try:
            det = self._backend.finish(job)
        except Exception as exc:  # noqa: BLE001
            self._fail(job, exc)
            return
        job.state.clear()  # 사진·텐서를 바로 놓는다
        try:
            job.future.set_result(det)
        except InvalidStateError:  # 이미 취소·종료됨
            pass

    @staticmethod
    def _fail(job: Job, exc: BaseException) -> None:
        job.state.clear()
        try:
            job.future.set_exception(exc)
        except InvalidStateError:  # 이미 끝났거나 취소됨
            pass

    def _take(self) -> tuple[str, list[Job]] | None:
        with self._cv:
            idle = False
            while not self._stopping and not any(self._queues.values()):
                idle = True
                self._cv.wait()
            if self._stopping:
                return None
            stage = min((s for s, q in self._queues.items() if q), key=lambda s: self._queues[s][0].enqueued)
            queue = self._queues[stage]
            limit = max(1, self._backend.max_batch(stage, self._max_batch))
            if idle and self._wait_s > 0:  # GPU 가 놀다가 첫 작업이 왔다 -> 잠깐 더 모은다
                deadline = queue[0].enqueued + self._wait_s
                while not self._stopping and len(queue) < limit and (rest := deadline - time.monotonic()) > 0:
                    self._cv.wait(rest)
                if self._stopping:
                    return None
            return stage, [queue.popleft() for _ in range(min(limit, len(queue)))]

    def _loop(self) -> None:
        while True:
            try:
                taken = self._take()
                if taken is None:
                    return
                stage, jobs = taken
            except Exception:  # noqa: BLE001 - 엔진 스레드는 죽지 않는다
                log.exception("엔진 대기열 오류")
                continue
            try:
                done = self._run(stage, jobs)
                with self._cv:
                    for job in done:
                        self._advance(job)
            except Exception as exc:  # noqa: BLE001 - 엔진 스레드는 죽지 않는다(그 묶음만 오류로 끝냄)
                log.exception("엔진 묶음 처리 오류")
                for job in jobs:
                    self._fail(job, exc)

    def _run(self, stage: str, jobs: list[Job]) -> list[Job]:
        """한 묶음 실행. 성공한 작업 목록을 돌려준다(실패한 작업은 여기서 오류로 끝낸다)."""
        t = time.perf_counter()
        try:
            self._backend.run_stage(stage, jobs)
            ok = jobs
        except Exception as exc:  # noqa: BLE001 - 묶음 실패 -> 한 장씩 다시
            if len(jobs) == 1:
                log.warning("단계 %s 실패 (프레임 건너뜀): %r", stage, exc)
                self._fail(jobs[0], exc)
                ok = []
            else:
                log.warning("단계 %s 묶음(%d장) 실패 -> 한 장씩 다시: %r", stage, len(jobs), exc)
                ok = []
                for job in jobs:
                    try:
                        self._backend.run_stage(stage, [job])
                        ok.append(job)
                    except Exception as one:  # noqa: BLE001
                        log.warning("단계 %s 실패 (프레임 건너뜀): %r", stage, one)
                        self._fail(job, one)
        self._timer.add(stage, time.perf_counter() - t, len(jobs))
        if self.batch_observer is not None:
            try:
                self.batch_observer(stage, len(jobs))
            except Exception:  # noqa: BLE001 - 지표 오류가 추론을 멈추지 않게
                pass
        return ok


# --- 가짜 백엔드(시험·모델 없는 부하 시험용) ---
_STATE = re.compile(rb"FAKE:([A-Z_]+)")


class FakeBatchBackend(BatchBackend):
    """FakeDetector 결과를 엔진 단계로 흉내 낸다. stage_ms: 단계별 묶음 1회 지연(ms), fail_keys: 그 키는 실패."""

    def __init__(self, scenario: str = "normal", *, stage_ms: dict[str, float] | None = None,
                 fail_keys: set[str] | None = None) -> None:
        from pair_vision.models.fake import FakeDetector

        self._fake = FakeDetector(scenario)
        self._stage_ms = stage_ms or {}
        self._fail = fail_keys or set()

    def prepare(self, job: Job) -> list[str]:
        return ["hands"] if job.frame.channel == PHONE else ["webcam"]

    def run_stage(self, stage: str, jobs: list[Job]) -> None:
        bad = [j.frame.key for j in jobs if j.frame.key in self._fail]
        if bad:
            raise RuntimeError(f"가짜 실패: {len(bad)}장")
        delay = self._stage_ms.get(stage, 0.0)
        if delay:
            time.sleep(delay / 1000)

    def finish(self, job: Job) -> Detection:
        return self._fake.run([job.frame])[0]
