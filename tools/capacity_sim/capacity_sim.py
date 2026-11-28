"""pAIr-vision 용량 시뮬레이터(이산 사건): 응시자 N명이 GPU 1장(또는 Pod K대)에서 실시간으로 처리되는지 추정한다.

서버 구조를 그대로 옮겼다(src/pair_vision/pipeline/worker.py · models/pool.py · session/mailbox.py · models/real.py):
- 응시자·채널마다 큐 길이 1: 처리 중에 새 프레임이 오면 옛 프레임을 버린다(replaced).
- 파이프라인은 "대기 중인 프레임 전부 꺼내기 -> 한 묶음으로 모델에 넘기기 -> 묶음이 전부 끝날 때까지 기다리기"를 되풀이한다
  (process_pending 이 take_all 뒤 detector.run_async 를 await). 느린 워커 하나가 다음 묶음을 늦춘다.
- 워커 풀: 응시자(키)마다 워커 고정 배정(가장 적게 맡은 워커). 워커 안에서는 프레임을 차례로 처리하고, GPU 는 워커들이 나눠 쓴다
  (GPU 를 FIFO 단일 서버로 근사: 프로세스 시간 분할의 문맥 전환 손실은 `ctx_overhead` 로).
- 휴대폰 프레임: 디코드 -> (D-FINE: dfine_interval_ms 주기) -> RTMW 손 -> (OWLv2: owl_interval_ms 주기). 웹캠 프레임: 디코드 -> YOLOX+RTMW.
  주기는 프레임의 recvMs(도착 시각) 기준이고 성공했을 때만 갱신(real.py 와 같다).
GPU 단계 시간은 가정값이다(README: D-FINE 약 97ms, OWLv2 약 155ms. RTMW·YOLOX 는 미측정 -> 범위). 결과는 "추정"이다.
실행: python capacity_sim.py [--out 결과.md]
"""

from __future__ import annotations

import argparse
import heapq
import math
import random
from dataclasses import dataclass, replace
from statistics import mean

MAX_BATCH = 16


@dataclass(frozen=True)
class Params:
    # --- GPU 단계 시간(ms, 장당, 배치 1) ---
    dfine_ms: float = 97.0      # README 실측
    owl_ms: float = 155.0       # README 실측
    rtmw_ms: float = 25.0       # 미측정(10~40 범위): 손(휴대폰: 프레임 전체 한 상자)
    yolox_ratio: float = 0.4    # 웹캠 사람 검출(YOLOX-tiny) = rtmw_ms 의 이 배수(가정)
    cpu_ms: float = 3.0         # 프레임당 CPU 기본(JPEG 디코드 약 1~2ms + 색 변환·RTMW/YOLOX 전처리·파이썬) - C 실측 기반
    dfine_cpu_ms: float = 10.5  # D-FINE 이미지 프로세서(C 실측 10.5ms), D-FINE 을 돌릴 때만
    owl_cpu_ms: float = 120.0   # OWLv2 프로세서(960 패딩·리사이즈, C 실측 120ms), OWLv2 를 돌릴 때만
    # --- 구조 ---
    dfine_interval_ms: float = 3000.0   # 0 이면 매 프레임
    owl_interval_ms: float = 5000.0
    # --- 개선 수단 ---
    fp16_speedup: float = 1.0           # D-FINE·OWLv2 에만(RTMW 는 이미 ONNX)
    batch: bool = False                 # 응시자 사이 묶음 추론(GPU 모델 프로세스 1개)
    batch_f8: float = 0.4               # 배치 8 이상에서 장당 비용 배수(1.0=이득 없음)
    cpu_par: int = 4                    # 묶음 추론 때 CPU 전처리 병렬도(스레드)
    workers: int = 6                    # 현재 구조의 워커 수(batch=False 일 때)
    ctx_overhead: float = 1.0           # 워커 여러 개가 GPU 를 나눌 때 GPU 시간 배수(1.0=손실 없음)
    pods: int = 1                       # Pod 대수(응시자를 균등 분배, 각 Pod 는 독립)
    # --- 구현한 새 구조(2026-11-27, VISION_BATCH=1): 연속 묶음 추론 ---
    stream: bool = False                # 단계별 대기열 + 엔진 스레드 1개 + 칸마다 처리 중 1장(끝나면 바로 다음)
    fast_pre_gpu: bool = False          # D-FINE·OWLv2 전처리를 GPU 로(CPU 비용 0, 대신 GPU 에 장당 몇 ms)
    owl_gpu_pre_ms: float = 3.0         # GPU 전처리 비용(OWLv2 960px, 가정)
    dfine_gpu_pre_ms: float = 1.0       # GPU 전처리 비용(D-FINE 640px, 가정)
    person_interval_ms: float = 3000.0  # 웹캠 YOLOX 사람 상자 재사용 주기(0=매 프레임)
    batch_overhead_ms: float = 2.0      # 묶음 1회마다 파이썬·후처리·동기화 고정 비용(가정)
    owl_max_batch: int = 4              # OWLv2 묶음 상한(GPU 메모리)
    wait_ms: float = 15.0               # GPU 가 놀 때 첫 작업 뒤 더 모으는 시간
    prep_threads: int = 4               # CPU 전처리 스레드
    all_gpu_pre: bool = False           # 전부 GPU(JPEG nvJPEG 풀기·RTMW 자르기·YOLOX 레터박스까지): CPU 는 단계 판단만
    plan_cpu_ms: float = 0.3            # all_gpu_pre 일 때 프레임당 CPU(단계 판단·파이썬)
    decode_gpu_ms: float = 1.0          # nvJPEG 풀기(장당, 가정) - 'decode' 단계로 묶음
    crop_gpu_ms: float = 0.3            # RTMW 자르기·YOLOX 레터박스(grid_sample, 장당, 가정)
    # --- 입력 ---
    fps: float = 1.0                    # 채널당 초당 장수
    channels: tuple[str, ...] = ("phone", "webcam")


@dataclass
class Result:
    drop_rate: float
    lat_p95_s: float
    lat_p50_s: float
    gap_viol_phone: float   # 휴대폰 채널: 처리된 프레임 간격 > 3.0초 비율
    gap_viol_webcam: float  # 웹캠 채널: 처리된 프레임 간격 > 2.5초 비율
    gap_p95_s: float
    gpu_util: float

    @property
    def gap_viol(self) -> float:
        return max(self.gap_viol_phone, self.gap_viol_webcam)


def f_batch(n: int, f8: float) -> float:
    """묶음 크기 n 일 때 장당 비용 배수: 1 -> n=8 에서 f8, 이후 평평(log2 보간)."""
    return 1.0 - (1.0 - f8) * min(1.0, math.log2(max(1, n)) / 3.0)


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


class OnePod:
    """Pod 한 대: n 명의 응시자를 시뮬레이션한다."""

    def __init__(self, p: Params, n: int, rng: random.Random) -> None:
        self.p, self.n, self.rng = p, n, rng
        self.dfine_ms = p.dfine_ms / p.fp16_speedup
        self.owl_ms = p.owl_ms / p.fp16_speedup
        self.hands_ms = p.rtmw_ms
        self.webcam_ms = p.rtmw_ms * (1 + p.yolox_ratio)
        self.worker_of: dict[int, int] = {}
        self.count = [0] * p.workers
        self.last_dfine: dict[int, float] = {}
        self.last_owl: dict[int, float] = {}
        self.gpu_busy = 0.0

    def assign(self, k: int) -> int:
        w = self.worker_of.get(k)
        if w is None:
            w = min(range(self.p.workers), key=lambda i: self.count[i])
            self.worker_of[k] = w
            self.count[w] += 1
        return w

    def needs(self, k: int, ch: str, t: float) -> tuple[bool, bool]:
        """(D-FINE 필요, OWLv2 필요) - 성공했을 때만 갱신하므로 여기서 갱신(실패 없음)."""
        if ch != "phone":
            return False, False
        p = self.p
        ms = t * 1000.0
        d = p.dfine_interval_ms <= 0 or k not in self.last_dfine or ms - self.last_dfine[k] >= p.dfine_interval_ms
        o = p.owl_interval_ms <= 0 or k not in self.last_owl or ms - self.last_owl[k] >= p.owl_interval_ms
        if d:
            self.last_dfine[k] = ms
        if o:
            self.last_owl[k] = ms
        return d, o

    def batch_time(self, frames: list[tuple[int, str, float]]) -> float:
        """한 묶음(대기 중이던 프레임 전부)이 끝나기까지 걸리는 시간(초)."""
        p = self.p
        plan = [(k, ch, *self.needs(k, ch, t)) for k, ch, t in frames]
        if p.batch:
            return self._batched(plan)
        return self._workers(plan)

    # --- 현재 구조: 워커 N개, GPU 시간 나눠 씀 ---
    def _workers(self, plan) -> float:
        p = self.p
        ops: dict[int, list[tuple[str, float]]] = {}
        for k, ch, d, o in plan:
            w = self.assign(k)
            seq = ops.setdefault(w, [])
            seq.append(("cpu", p.cpu_ms))
            if ch == "phone":
                if d:
                    seq += [("cpu", p.dfine_cpu_ms), ("gpu", self.dfine_ms)]
                seq.append(("gpu", self.hands_ms))
                if o:
                    seq += [("cpu", p.owl_cpu_ms), ("gpu", self.owl_ms)]
            else:
                seq.append(("gpu", self.webcam_ms))
        mult = p.ctx_overhead if len(ops) > 1 else 1.0
        heap = [(0.0, w, 0) for w in ops]
        heapq.heapify(heap)
        gpu_free, end = 0.0, 0.0
        while heap:
            t, w, i = heapq.heappop(heap)
            if i >= len(ops[w]):
                end = max(end, t)
                continue
            kind, ms = ops[w][i]
            if kind == "cpu":
                heapq.heappush(heap, (t + ms / 1000, w, i + 1))
            else:
                start = max(t, gpu_free)
                gpu_free = start + ms * mult / 1000
                self.gpu_busy += ms * mult / 1000
                heapq.heappush(heap, (gpu_free, w, i + 1))
        return end

    # --- 묶음 추론: 모델 프로세스 1개가 단계별로 응시자들을 묶어 처리 ---
    def _batched(self, plan) -> float:
        p = self.p
        cpu = sum(p.cpu_ms + (p.dfine_cpu_ms if x[2] else 0) + (p.owl_cpu_ms if x[3] else 0) for x in plan)
        total = cpu / p.cpu_par / 1000

        def stage(count: int, ms: float) -> float:
            t = 0.0
            while count > 0:
                m = min(count, MAX_BATCH)
                t += m * ms * f_batch(m, p.batch_f8) / 1000
                count -= m
            return t

        phone = [x for x in plan if x[1] == "phone"]
        gpu = (
            stage(sum(1 for x in phone if x[2]), self.dfine_ms)
            + stage(len(phone), self.hands_ms)
            + stage(sum(1 for x in phone if x[3]), self.owl_ms)
            + stage(len(plan) - len(phone), self.webcam_ms)
        )
        self.gpu_busy += gpu
        return total + gpu


def simulate_pod(p: Params, n: int, duration_s: float, warmup_s: float, rng: random.Random):
    pod = OnePod(p, n, rng)
    # 도착 생성: 응시자·채널마다 위상 무작위, 주기 1/fps (+-20ms 지터)
    arrivals: list[tuple[float, int, str]] = []
    period = 1.0 / p.fps
    for k in range(n):
        for ch in p.channels:
            t = rng.random() * period
            while t < duration_s:
                arrivals.append((t, k, ch))
                t += period + rng.uniform(-0.02, 0.02)
    arrivals.sort()
    mailbox: dict[tuple[int, str], float] = {}
    lat: list[float] = []
    last_proc: dict[tuple[int, str], float] = {}
    gaps = {"phone": [], "webcam": []}
    received = dropped = 0
    i, now = 0, 0.0
    while i < len(arrivals) or mailbox:
        if not mailbox:
            now = max(now, arrivals[i][0])
        while i < len(arrivals) and arrivals[i][0] <= now:
            t, k, ch = arrivals[i]
            i += 1
            if t >= warmup_s:
                received += 1
                dropped += (k, ch) in mailbox
            mailbox[(k, ch)] = t  # 같은 칸이 있으면 옛 프레임을 버리고 최신으로 교체
        frames = [(k, ch, t) for (k, ch), t in mailbox.items()]
        mailbox = {}
        end = now + pod.batch_time(frames)
        for k, ch, t in frames:
            if t >= warmup_s:
                lat.append(end - t)
                prev = last_proc.get((k, ch))
                if prev is not None:
                    gaps[ch].append(t - prev)
            last_proc[(k, ch)] = t
        now = end
    return received, dropped, lat, gaps, pod.gpu_busy / max(1e-9, duration_s)


def simulate_stream(p: Params, n: int, duration_s: float, warmup_s: float, rng: random.Random):
    """새 구조(src/pair_vision/models/batch.py + pipeline 연속 처리)의 이산 사건 시뮬레이션.

    - 칸(응시자·채널)마다 대기 1장 + 처리 중 1장. 처리 중에 새 프레임이 두 번 오면 앞 것을 버린다(replaced).
    - CPU 전처리 스레드 prep_threads 개(디코드 + 기준 전처리일 때 D-FINE·OWLv2 전처리)
    - GPU 엔진 1개: 가장 오래 기다린 작업이 있는 단계를 골라 그 단계 대기열에서 최대 MAX_BATCH(OWLv2 는 owl_max_batch)장.
      비용 = 묶음 고정비 + 장 수 x 장당 ms x f_batch(장 수). person(YOLOX) 은 ONNX 배치 1 고정이라 이득 없음.
      휴대폰 손과 웹캠 얼굴·몸은 같은 rtmw 단계에 섞인다. GPU 가 놀다가 일이 오면 wait_ms 만큼 모은다.
    """
    dfine_ms = p.dfine_ms / p.fp16_speedup + (p.dfine_gpu_pre_ms if p.fast_pre_gpu else 0.0)
    owl_ms = p.owl_ms / p.fp16_speedup + (p.owl_gpu_pre_ms if p.fast_pre_gpu else 0.0)
    person_ms = p.rtmw_ms * p.yolox_ratio
    period = 1.0 / p.fps
    events: list = []
    seq = 0

    def push(t, kind, data):
        nonlocal seq
        seq += 1
        heapq.heappush(events, (t, seq, kind, data))

    for k in range(n):
        for ch in p.channels:
            t = rng.random() * period
            while t < duration_s:
                push(t, "arrive", (k, ch, t))
                t += period + rng.uniform(-0.02, 0.02)
    mailbox: dict = {}
    busy: set = set()
    last = {"dfine": {}, "owl": {}, "person": {}}
    prep_free = [0.0] * p.prep_threads
    queues: dict[str, list] = {"decode": [], "dfine": [], "person": [], "rtmw": [], "owlv2": []}
    gpu_busy_until = -1.0
    gpu_pending = False
    gpu_time = 0.0
    received = dropped = 0
    lat: list[float] = []
    gaps = {"phone": [], "webcam": []}
    last_proc: dict = {}

    def due(table, k, ms, interval):
        return interval <= 0 or k not in table or ms - table[k] >= interval

    def dispatch(slot, now):
        t_arr = mailbox.pop(slot)
        busy.add(slot)
        k, ch = slot
        ms = t_arr * 1000.0
        stages, cpu = (["decode"], p.plan_cpu_ms) if p.all_gpu_pre else ([], p.cpu_ms)
        if ch == "phone":
            if due(last["dfine"], k, ms, p.dfine_interval_ms):
                last["dfine"][k] = ms
                stages.append("dfine")
                cpu += 0.0 if p.fast_pre_gpu else p.dfine_cpu_ms
            stages.append("rtmw")
            if due(last["owl"], k, ms, p.owl_interval_ms):
                last["owl"][k] = ms
                stages.append("owlv2")
                cpu += 0.0 if p.fast_pre_gpu else p.owl_cpu_ms
        else:
            if due(last["person"], k, ms, p.person_interval_ms):
                last["person"][k] = ms
                stages.append("person")
            stages.append("rtmw")
        i = min(range(len(prep_free)), key=lambda j: prep_free[j])
        start = max(now, prep_free[i])
        prep_free[i] = start + cpu / 1000
        push(prep_free[i], "prepped", {"slot": slot, "t": t_arr, "stages": stages})

    def enqueue(job, now):
        nonlocal gpu_pending
        stage = job["stages"].pop(0)
        job["enq"] = now
        queues[stage].append(job)
        if now >= gpu_busy_until and not gpu_pending:  # GPU 가 놀고 있었다 -> wait_ms 모은 뒤 시작
            gpu_pending = True
            push(now + p.wait_ms / 1000, "gpu", None)

    def gpu_start(now):
        nonlocal gpu_busy_until, gpu_time
        heads = [s for s, q in queues.items() if q]
        if not heads:
            return
        stage = min(heads, key=lambda s: queues[s][0]["enq"])
        limit = p.owl_max_batch if stage == "owlv2" else MAX_BATCH
        jobs, queues[stage][:] = queues[stage][:limit], queues[stage][limit:]
        m = len(jobs)
        crop = p.crop_gpu_ms if p.all_gpu_pre else 0.0
        unit = {"decode": p.decode_gpu_ms, "dfine": dfine_ms, "owlv2": owl_ms, "rtmw": p.rtmw_ms + crop,
                "person": person_ms + crop}[stage]
        factor = 1.0 if stage == "person" else f_batch(m, p.batch_f8)
        dur = (p.batch_overhead_ms + m * unit * factor) / 1000
        gpu_time += dur
        gpu_busy_until = now + dur
        push(gpu_busy_until, "gpu_done", jobs)

    while events:
        now, _, kind, data = heapq.heappop(events)
        if kind == "arrive":
            k, ch, t = data
            slot = (k, ch)
            if t >= warmup_s:
                received += 1
                dropped += slot in mailbox
            mailbox[slot] = t
            if slot not in busy:
                dispatch(slot, now)
        elif kind == "prepped":
            enqueue(data, now)
        elif kind == "gpu":
            gpu_pending = False
            if now >= gpu_busy_until:
                gpu_start(now)
        elif kind == "gpu_done":
            for job in data:
                if job["stages"]:
                    stage = job["stages"].pop(0)
                    job["enq"] = now
                    queues[stage].append(job)
                    continue
                slot, t_arr = job["slot"], job["t"]
                if t_arr >= warmup_s:
                    lat.append(now - t_arr)
                    prev = last_proc.get(slot)
                    if prev is not None:
                        gaps[slot[1]].append(t_arr - prev)
                last_proc[slot] = t_arr
                busy.discard(slot)
                if slot in mailbox:
                    dispatch(slot, now)
            gpu_start(now)  # 바쁘게 이어서(대기 없이)
    return received, dropped, lat, gaps, gpu_time / max(1e-9, duration_s)


def simulate(p: Params, n_total: int, *, duration_s: float = 240.0, warmup_s: float = 30.0, seeds=(1, 2, 3)) -> Result:
    recv = drop = 0
    lat: list[float] = []
    gaps = {"phone": [], "webcam": []}
    utils = []
    for seed in seeds:
        for pod in range(p.pods):
            n = n_total // p.pods + (1 if pod < n_total % p.pods else 0)
            if n == 0:
                continue
            sim = simulate_stream if p.stream else simulate_pod
            r, d, la, ga, u = sim(p, n, duration_s, warmup_s, random.Random(seed * 1000 + pod))
            recv += r
            drop += d
            lat += la
            for ch in gaps:
                gaps[ch] += ga[ch]
            utils.append(u)
    viol = lambda xs, lim: (sum(1 for g in xs if g > lim) / len(xs)) if xs else 0.0  # noqa: E731
    allg = gaps["phone"] + gaps["webcam"]
    return Result(
        drop_rate=drop / max(1, recv),
        lat_p95_s=percentile(lat, 0.95),
        lat_p50_s=percentile(lat, 0.5),
        gap_viol_phone=viol(gaps["phone"], 3.0),
        gap_viol_webcam=viol(gaps["webcam"], 2.5),
        gap_p95_s=percentile(allg, 0.95),
        gpu_util=mean(utils) if utils else 0.0,
    )


# "문제없음" 판정: 끊김 허용 초과 간격 비율 <= 1%, 프레임 폐기 <= 10%, 지연 p95 <= 3초
def ok(r: Result) -> bool:
    return r.gap_viol <= 0.01 and r.drop_rate <= 0.10 and r.lat_p95_s <= 3.0


# GPU 시험 합격 기준(docs/intent/realtime-vision-20.md): 폐기 <= 1%, 지연 p95 <= 3초, 끊김 허용 초과 <= 1%
def ok_strict(r: Result) -> bool:
    return r.gap_viol <= 0.01 and r.drop_rate <= 0.01 and r.lat_p95_s <= 3.0


def capacity(p: Params, hi: int = 60, judge=ok) -> int:
    """판정(judge)을 만족하는 최대 응시자 수(1 Pod 기준 Params.pods 대 전체)."""
    best = 0
    for n in range(1, hi + 1):
        if judge(simulate(p, n, duration_s=150, warmup_s=20, seeds=(1, 2))):
            best = n
        elif n > best + 3:  # 한 번 깨진 뒤 3명 더 가도 안 되면 중단
            break
    return best


def scenarios(base: Params) -> list[tuple[str, Params]]:
    fp = 1.75
    return [
        ("① 현재 구조(워커6, D-FINE 3초·OWL 5초)", replace(base, workers=6)),
        ("② fp16(D-FINE·OWL x1.75)", replace(base, workers=6, fp16_speedup=fp)),
        ("③a OWL 10초", replace(base, workers=6, owl_interval_ms=10000)),
        ("③b OWL 15초", replace(base, workers=6, owl_interval_ms=15000)),
        ("④ 응시자 묶음 추론(모델 1프로세스)", replace(base, batch=True)),
        ("⑤a Pod 2대(각 ①)", replace(base, workers=6, pods=2)),
        ("⑤b Pod 3대(각 ①)", replace(base, workers=6, pods=3)),
        ("⑥ ②+③a+④", replace(base, batch=True, fp16_speedup=fp, owl_interval_ms=10000)),
        ("⑥' ②+③b+④", replace(base, batch=True, fp16_speedup=fp, owl_interval_ms=15000)),
    ]


def gpus_needed(p: Params, n: int, cap: int = 8) -> int | None:
    """n 명을 판정(ok)을 만족하며 처리하는 데 필요한 최소 Pod(GPU) 수. cap 대로도 안 되면 None."""
    for k in range(1, cap + 1):
        if ok(simulate(replace(p, pods=k), n, duration_s=150, warmup_s=20, seeds=(1, 2))):
            return k
    return None


def row(n: int, r: Result) -> str:
    return (f"| {n} | {r.drop_rate:.1%} | {r.lat_p95_s:.2f} | {r.gap_viol_phone:.1%} / {r.gap_viol_webcam:.1%} | "
            f"{r.gpu_util:.0%} | {'OK' if ok(r) else '부족'} |")


def new_structure(base: Params, **kw) -> Params:
    """구현한 구조(VISION_BATCH=1, rules 2026-11-27.1): 연속 묶음 + fp16 + OWLv2 10초 + GPU 전처리 + 사람 상자 3초 재사용."""
    return replace(base, stream=True, batch=True, fp16_speedup=kw.pop("fp16_speedup", 1.75), owl_interval_ms=10000,
                   fast_pre_gpu=True, all_gpu_pre=kw.pop("all_gpu_pre", True), person_interval_ms=3000, **kw)


def main_new(base: Params) -> None:
    H = "| 응시자 | 폐기율 | 지연 p95(s) | 끊김 허용 초과 간격(폰/웹캠) | GPU 점유 | 판정(시뮬 기준) | 합격 기준(폐기<=1%) |\n|---|---|---|---|---|---|---|"
    print("## ⑦ 구현한 구조(연속 묶음 추론 + fp16 + OWLv2 10초 + GPU 전처리 + 사람 상자 재사용)")
    p7 = new_structure(base)
    print(H)
    for n in (10, 15, 20, 25, 30, 35):
        r = simulate(p7, n)
        print(row(n, r) + f" {'합격' if ok_strict(r) else '불합격'} |")
    print("\n## 비교: 묶음 동기식(⑥, 예전 시뮬) vs 연속(⑦), 같은 가정")
    print("| 구조 | 20명 폐기율 | 20명 지연 p95 | GPU 1장 최대(폐기<=10%) | GPU 1장 최대(합격 기준, 폐기<=1%) |\n|---|---|---|---|---|")
    p6 = replace(base, batch=True, fp16_speedup=1.75, owl_interval_ms=10000)
    for name, p in (("⑥ 묶음 동기식", p6), ("⑦ 연속 묶음(구현)", p7)):
        r = simulate(p, 20)
        print(f"| {name} | {r.drop_rate:.1%} | {r.lat_p95_s:.2f} | {capacity(p)} | {capacity(p, judge=ok_strict)} |")
    print("\n## ⑦ 민감도: GPU 1장 최대 인원(합격 기준) / 20명 결과")
    print("| 가정 변경 | 최대 인원 | 20명 폐기율 | 20명 지연 p95 | 20명 GPU 점유 | 20명 판정 |\n|---|---|---|---|---|---|")
    cases = [
        ("기준(RTMW 25ms, f8=0.4, fp16 1.75x, 묶음 고정비 2ms)", {}),
        ("RTMW·YOLOX 10ms", {"rtmw_ms": 10.0}),
        ("RTMW·YOLOX 40ms", {"rtmw_ms": 40.0}),
        ("RTMW·YOLOX 60ms", {"rtmw_ms": 60.0}),
        ("배치 이득 약함 f8=0.5", {"batch_f8": 0.5}),
        ("배치 이득 큼 f8=0.3", {"batch_f8": 0.3}),
        ("배치 이득 없음 f8=1.0", {"batch_f8": 1.0}),
        ("fp16 이득 없음(1.0)", {"fp16_speedup": 1.0}),
        ("fp16 2배", {"fp16_speedup": 2.0}),
        ("README 값에 전처리 포함(D-FINE 87·OWL 35ms)", {"dfine_ms": 87.0, "owl_ms": 35.0}),
        ("묶음 고정비 8ms", {"batch_overhead_ms": 8.0}),
        ("GPU 전처리 OWL 10ms", {"owl_gpu_pre_ms": 10.0}),
        ("배치 이득 없음 + RTMW 40ms(최악 조합)", {"batch_f8": 1.0, "rtmw_ms": 40.0}),
        ("JPEG 풀기·자르기는 CPU(전부 GPU 이전)", {"all_gpu_pre": False}),
        ("GPU 풀기 3ms·자르기 1ms(느림)", {"decode_gpu_ms": 3.0, "crop_gpu_ms": 1.0}),
        ("CPU 전처리 스레드 1개 + Pod CPU 느림(전부 GPU 이전, cpu 3배)", {"all_gpu_pre": False, "prep_threads": 1, "cpu_ms": 9.0}),
    ]
    for label, kw in cases:
        p = new_structure(base, **kw)
        r = simulate(p, 20)
        print(f"| {label} | {capacity(p, judge=ok_strict)} | {r.drop_rate:.1%} | {r.lat_p95_s:.2f} | {r.gpu_util:.0%} | "
              f"{'합격' if ok_strict(r) else '불합격'} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtmw", type=float, default=25.0)
    ap.add_argument("--cpu", type=float, default=3.0)
    ap.add_argument("--ctx", type=float, default=1.0)
    ap.add_argument("--new-only", action="store_true", help="구현한 새 구조(⑦)만")
    args = ap.parse_args()
    base = Params(rtmw_ms=args.rtmw, cpu_ms=args.cpu, ctx_overhead=args.ctx)
    if args.new_only:
        main_new(base)
        return
    H = "| 응시자 | 폐기율 | 지연 p95(s) | 끊김 허용 초과 간격(폰/웹캠) | GPU 점유 | 판정 |\n|---|---|---|---|---|---|"

    print("## 보정: 현재 구조(워커 6), D-FINE 매 프레임(주기화 전), OWL 5초  (README 실측: 4명 OK, 8명 부족)")
    old = replace(base, workers=6, dfine_interval_ms=0)
    print(H)
    for n in (2, 4, 5, 6, 7, 8, 10):
        print(row(n, simulate(old, n)))
    print(f"-> 최대 {capacity(old, 20)}명")

    print("\n## 시나리오별(기준 가정값, 1 Pod 단위. ⑤는 Pod 대수만큼 응시자를 나눔)")
    for name, p in scenarios(base):
        print(f"\n### {name}\n{H}")
        for n in (5, 10, 15, 20, 25, 30):
            print(row(n, simulate(p, n)))

    print("\n## 요약: GPU 1장(1 Pod)당 최대 인원 / 20명에 필요한 GPU 수  (판정: 끊김초과<=1%, 폐기<=10%, 지연p95<=3s)")
    print("| 시나리오 | GPU 1장당 최대 인원 | 20명에 필요한 GPU 수 | 30명에 필요한 GPU 수 |\n|---|---|---|---|")
    for name, p in scenarios(base):
        if p.pods > 1:
            continue
        print(f"| {name} | {capacity(p)} | {gpus_needed(p, 20)} | {gpus_needed(p, 30)} |")

    print("\n## 민감도(GPU 1장당 최대 인원)")
    print("| 가정 변경 | ① 현재 | ② fp16 | ③b OWL15초 | ④ 묶음 | ⑥ ②+③a+④ |\n|---|---|---|---|---|---|")
    cases = [
        ("기준(RTMW 25ms, 배치 f8=0.4, fp16 1.75x)", {}),
        ("RTMW·YOLOX 10ms(빠름)", {"rtmw_ms": 10.0}),
        ("RTMW·YOLOX 40ms(느림)", {"rtmw_ms": 40.0}),
        ("배치 이득 약함 f8=0.5", {"batch_f8": 0.5}),
        ("배치 이득 큼 f8=0.3", {"batch_f8": 0.3}),
        ("배치 이득 없음 f8=1.0", {"batch_f8": 1.0}),
        ("fp16 1.5배", {"fp16_speedup": 1.5}),
        ("fp16 2배", {"fp16_speedup": 2.0}),
        ("워커 여러 개 GPU 시분할 손실 +15%", {"ctx_overhead": 1.15}),
        ("README 값에 전처리 포함(D-FINE GPU 87·OWL GPU 35ms)", {"dfine_ms": 87.0, "owl_ms": 35.0}),
        ("OWL 전처리 CPU 120ms 를 GPU 로 옮김(0ms)", {"owl_cpu_ms": 0.0}),
    ]
    for label, kw in cases:
        b = replace(base, **{k: v for k, v in kw.items() if k != "fp16_speedup"})
        fp = kw.get("fp16_speedup", 1.75)
        sc = {
            1: replace(b, workers=6),
            2: replace(b, workers=6, fp16_speedup=fp),
            3: replace(b, workers=6, owl_interval_ms=15000),
            4: replace(b, batch=True),
            5: replace(b, batch=True, fp16_speedup=fp, owl_interval_ms=10000),
        }
        print(f"| {label} | " + " | ".join(str(capacity(sc[i])) for i in (1, 2, 3, 4, 5)) + " |")
    print()
    main_new(base)


if __name__ == "__main__":
    main()
