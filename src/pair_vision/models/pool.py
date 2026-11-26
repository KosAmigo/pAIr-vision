"""프로세스 풀 Detector: 모델을 프로세스 여러 개에 올려 프레임을 병렬로 분석한다.

실측(RTX A5000 Pod): 한 프로세스가 프레임을 한 장씩 처리해 초당 약 5장에서 막히고, GPU 는 3~17% 만 쓰며 CPU load 만 높았다
(손·얼굴·포즈 검출이 CPU, 스레드 경쟁). 그래서 모델을 프로세스마다 따로 올리고:
- 응시자(visionKey) 하나의 프레임은 항상 같은 프로세스가 맡는다(고정 배정). OWLv2 주기·웹캠 포즈 캐시 같은 세션별 상태가 한 곳에 있고 순서가 지켜진다.
- 새 응시자는 맡은 응시자가 가장 적은 프로세스에 배정한다(균등 분배).
- 프로세스마다 쓸 CPU 코어를 나눠 고정하고(리눅스) 수치 라이브러리 스레드 수를 맞춰 스레드 경쟁을 막는다.
프로세스가 죽으면 그 프로세스만 다시 띄운다(그동안 그 응시자의 프레임은 오류로 버려지고 서버는 계속 돈다).
세션 상태(규칙 엔진)는 메인 프로세스에만 있고, 워커는 프레임 -> Detection 변환만 한다.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import threading
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from pair_vision.models.base import Detection, Detector, Frame

log = logging.getLogger("pair_vision.pool")

_DETECTOR: Detector | None = None


def _init_worker(spec: dict, cpus: list[int], threads: int) -> None:
    """워커 프로세스 시작: 코어·스레드 수를 정한 뒤(모델 import 전에) 모델을 적재한다."""
    global _DETECTOR
    threads = max(1, threads)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)
    if cpus and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, set(cpus))
        except OSError:
            pass
    from pair_vision.models.fake import create_detector

    _DETECTOR = create_detector(**spec)
    _DETECTOR.load()
    if spec["kind"] == "real":
        import torch

        torch.set_num_threads(threads)


def _info() -> dict:
    assert _DETECTOR is not None
    return {"gpu": _DETECTOR.gpu, "name": _DETECTOR.name}


def _run(frames: list[Frame]) -> list[Detection]:
    assert _DETECTOR is not None
    return _DETECTOR.run(frames)


def _stats() -> dict:
    return _DETECTOR.stats() if _DETECTOR is not None else {}


def _forget(key: str) -> None:
    if _DETECTOR is not None:
        _DETECTOR.forget(key)


class PooledDetector(Detector):
    name = "pool"

    def __init__(self, spec: dict, workers: int, threads: int = 0) -> None:
        """spec: create_detector(**spec) 로 워커가 만들 Detector 설정(kind·scenario·models_dir 등, workers 는 넣지 않는다)."""
        if workers < 2:
            raise ValueError("PooledDetector 는 워커 2개 이상에서 쓴다")
        self._spec = {**spec, "workers": 1, "worker_threads": 0}
        self._n = workers
        self.gpu = False
        cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        # 워커마다 쓸 코어 수: 지정값 > (사용 가능 코어 / 워커 수). 코어 목록을 못 얻는 OS(윈도우)는 전체 코어 수로 나눈다.
        per = threads or max(1, (len(cpus) or os.cpu_count() or 1) // workers)
        self._threads = per
        self._slices = [[cpus[(i * per + j) % len(cpus)] for j in range(per)] if cpus else [] for i in range(workers)]
        self._ctx = multiprocessing.get_context("spawn")  # CUDA 는 fork 와 같이 쓸 수 없다
        self._execs: list[ProcessPoolExecutor] = []
        self._assign: dict[str, int] = {}
        self._count = [0] * workers
        self._lock = threading.Lock()

    # --- 수명 ---
    def _make(self, index: int) -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=1, mp_context=self._ctx, initializer=_init_worker, initargs=(self._spec, self._slices[index], self._threads)
        )

    def load(self) -> None:
        self._execs = [self._make(i) for i in range(self._n)]
        infos = [f.result() for f in [ex.submit(_info) for ex in self._execs]]  # 모든 워커가 모델을 올릴 때까지
        self.gpu = any(i["gpu"] for i in infos)
        log.info("워커 %d개 적재 완료 (코어 배정 %s)", self._n, self._slices)

    def stats(self) -> dict[str, tuple[float, float]]:
        """모든 워커의 단계별 (호출 수, 누적 초)를 합친다."""
        merged: dict[str, list[float]] = {}
        for ex in self._execs:
            try:
                part = ex.submit(_stats).result(timeout=5)
            except Exception:  # noqa: BLE001 - 워커가 죽었거나 바쁘면 이번 집계에서 뺀다
                continue
            for stage, (calls, seconds) in part.items():
                item = merged.setdefault(stage, [0.0, 0.0])
                item[0] += calls
                item[1] += seconds
        return {k: (v[0], v[1]) for k, v in merged.items()}

    def close(self) -> None:
        for ex in self._execs:
            ex.shutdown(wait=False, cancel_futures=True)

    def _respawn(self, index: int) -> None:
        log.error("워커 %d 가 죽어 다시 띄웁니다", index)
        old = self._execs[index]
        self._execs[index] = self._make(index)
        old.shutdown(wait=False, cancel_futures=True)

    # --- 배정 ---
    def worker_for(self, key: str) -> int:
        with self._lock:
            idx = self._assign.get(key)
            if idx is None:
                idx = min(range(self._n), key=lambda i: self._count[i])
                self._assign[key] = idx
                self._count[idx] += 1
            return idx

    def forget(self, key: str) -> None:
        with self._lock:
            idx = self._assign.pop(key, None)
            if idx is not None:
                self._count[idx] -= 1
        if idx is not None and self._execs:
            try:
                self._execs[idx].submit(_forget, key)
            except Exception:  # noqa: BLE001 - 종료 중이거나 워커가 죽은 경우
                pass

    # --- 실행 ---
    def _submit(self, frames: Sequence[Frame]) -> list[tuple[int, Future]]:
        groups: dict[int, list[Frame]] = defaultdict(list)
        for frame in frames:
            groups[self.worker_for(frame.key)].append(frame)
        return [(i, self._execs[i].submit(_run, fs)) for i, fs in groups.items()]

    def run(self, frames: Sequence[Frame]) -> list[Detection]:
        out: list[Detection] = []
        for i, fut in self._submit(frames):
            try:
                out.extend(fut.result())
            except BrokenProcessPool:
                self._respawn(i)
                raise
        return out

    async def run_async(self, frames: Sequence[Frame]) -> list[Detection]:
        submitted = self._submit(frames)
        results = await asyncio.gather(*(asyncio.wrap_future(f) for _, f in submitted), return_exceptions=True)
        out: list[Detection] = []
        first_error: BaseException | None = None
        for (i, _), res in zip(submitted, results, strict=True):
            if isinstance(res, BaseException):
                first_error = first_error or res
                if isinstance(res, BrokenProcessPool):
                    self._respawn(i)
                else:
                    log.error("워커 %d 분석 오류: %r", i, res)
            else:
                out.extend(res)
        if first_error is not None and not out:
            raise first_error
        return out
