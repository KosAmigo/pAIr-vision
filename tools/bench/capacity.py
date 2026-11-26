"""용량 측정: 네트워크 없이 가상 응시자 N명이 웹캠·휴대폰 사진을 초당 1장씩 보낼 때, 몇 명까지 감당하는지 잰다.

서버와 같은 경로를 쓴다: 응시자·채널마다 큐 길이 1(최신만, 밀리면 버림) -> Detector(.run_async) / 워커 풀.
규칙 엔진·WebSocket 은 빼고 모델 처리만 본다(그쪽은 거의 비용이 없다).

  # 가짜 모델로 동작 확인(몇 초)
  python -m tools.bench.capacity --detector fake --workers 1 --batch off --fp16 off --webcam server --examinees 5,10 --duration 5

  # 실제 모델 전체 조합(RunPod): 조합이 많아 1~2시간 걸린다. 일부만 고르려면 목록을 줄인다.
  python -m tools.bench.capacity --detector real --photos photos --out /workspace/capacity

스레드: --worker-threads(0=자동=실효 CPU ÷ workers). 실효 CPU 는 cgroup 한도·affinity 를 반영한다(os.cpu_count() 가 아님).
조합 = workers {1,2,4,8} x 배치 {off,on} x fp16 {off,on} x 웹캠 {server,off} x 응시자 {10,20,30,40}. 각 인자는 쉼표 목록이다.
결과: <out>.csv, <out>.md (맨 위에 "조합별로 응시자당 0.9fps 이상을 유지한 최대 인원").
사용 가능한 환경이면 GPU(pynvml 또는 nvidia-smi)·CPU(psutil) 사용률도 같이 적는다. 없으면 그 칸만 비운다.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import os
import platform
import shutil
import subprocess
import threading
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pair_vision.models.base import PHONE, WEBCAM, Detector, Frame
from pair_vision.models.cpuinfo import cpu_info, default_threads, resolve_threads
from pair_vision.models.fake import create_detector
from pair_vision.models.options import RunOptions
from pair_vision.session.frames import ParsedFrame
from pair_vision.session.mailbox import Mailboxes

from .harness import IMAGE_SUFFIXES, percentile

STAGES = ("decode", "dfine", "owl", "hands", "face", "pose")
PASS_FPS = 0.9
DEFAULT_WORKERS = "1,2,4,8"
DEFAULT_EXAMINEES = "10,20,30,40"


# --- 조합 ---
@dataclass(frozen=True)
class Combo:
    workers: int
    batch: bool
    fp16: bool
    webcam: bool  # True=server(서버가 웹캠 분석), False=off
    threads: int = 0  # 워커당 스레드 수. 0=자동(실효 CPU ÷ 워커 수)

    def options(self) -> RunOptions:
        return RunOptions(batch_gpu=self.batch, fp16=self.fp16, webcam=self.webcam)

    def key(self) -> tuple:
        return (self.workers, self.batch, self.fp16, self.webcam, self.threads)

    def label(self) -> str:
        return (
            f"workers={self.workers} threads={self.threads or 'auto'} batch={'on' if self.batch else 'off'} "
            f"fp16={'on' if self.fp16 else 'off'} webcam={'server' if self.webcam else 'off'}"
        )


def parse_ints(text: str) -> list[int]:
    values = [int(v) for v in text.split(",") if v.strip()]
    if not values or any(v < 1 for v in values):
        raise ValueError(f"1 이상의 정수 목록이어야 합니다: {text!r}")
    return values


def parse_choices(text: str, names: tuple[str, str]) -> list[bool]:
    """'off,on' 같은 목록 -> [False, True]. names=(거짓 이름, 참 이름)."""
    out = []
    for part in (p.strip().lower() for p in text.split(",") if p.strip()):
        if part not in names:
            raise ValueError(f"{names[0]} 또는 {names[1]} 여야 합니다: {part!r}")
        out.append(part == names[1])
    if not out:
        raise ValueError("값이 비었습니다")
    return list(dict.fromkeys(out))


def parse_threads(text: str) -> list[int]:
    """워커당 스레드 목록. 0=자동."""
    values = [int(v) for v in text.split(",") if v.strip()]
    if not values or any(v < 0 for v in values):
        raise ValueError(f"0 이상의 정수 목록이어야 합니다(0=자동): {text!r}")
    return list(dict.fromkeys(values))


def build_combos(workers: Iterable[int], batch: Iterable[bool], fp16: Iterable[bool], webcam: Iterable[bool],
                 threads: Iterable[int] = (0,)) -> list[Combo]:
    return [Combo(w, b, f, c, t) for w, b, f, c, t in itertools.product(workers, batch, fp16, webcam, threads)]


# --- 사진 ---
def make_photo_set(folder: Path | None, phone_edge: int, webcam_edge: int, phone_q: int, webcam_q: int) -> tuple[list[bytes], list[bytes], str]:
    """사진 폴더를 채널별 크기·화질 JPEG 로 미리 만들어 둔다. 폴더가 없으면 속도 측정용 합성 사진."""
    import io

    from PIL import Image

    images: list[Image.Image] = []
    note = f"사진 폴더 {folder}"
    if folder is not None and folder.is_dir():
        paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        images = [Image.open(p).convert("RGB") for p in paths]
    if not images:
        import numpy as np

        rng = np.random.default_rng(0)
        images = [Image.fromarray((rng.random((720, 960, 3)) * 80 + 100).astype("uint8")) for _ in range(8)]
        note = "합성 사진(폴더 없음: 속도만 의미 있고 검출 결과는 의미 없음)"

    def encode(img: Image.Image, edge: int, quality: int) -> bytes:
        copy = img.copy()
        copy.thumbnail((edge, edge))
        buf = io.BytesIO()
        copy.save(buf, "JPEG", quality=quality)
        return buf.getvalue()

    return [encode(i, phone_edge, phone_q) for i in images], [encode(i, webcam_edge, webcam_q) for i in images], note


# --- 하드웨어 사용률 ---
class Sampler(threading.Thread):
    """0.5초마다 GPU·CPU 사용률을 읽는다. 읽을 수 없는 것은 비워 둔다."""

    def __init__(self, interval: float = 0.5) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self._halt = threading.Event()
        self.gpu_util: list[float] = []
        self.gpu_mem: list[float] = []
        self.cpu: list[float] = []
        self.gpu_name = ""
        self._nvml = None
        self._smi = shutil.which("nvidia-smi")
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
            name = pynvml.nvmlDeviceGetName(self._nvml[1])
            self.gpu_name = name.decode() if isinstance(name, bytes) else str(name)
        except Exception:  # noqa: BLE001 - pynvml 없음: nvidia-smi 로
            self._nvml = None
        try:
            import psutil

            self._psutil = psutil
            psutil.cpu_percent(interval=None)
        except ImportError:
            self._psutil = None

    def _read_gpu(self) -> tuple[float, float] | None:
        if self._nvml:
            lib, handle = self._nvml
            util = lib.nvmlDeviceGetUtilizationRates(handle).gpu
            mem = lib.nvmlDeviceGetMemoryInfo(handle).used / 1048576
            return float(util), float(mem)
        if self._smi:
            out = subprocess.run(
                [self._smi, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=False,
            ).stdout.strip().splitlines()
            if out:
                util, mem = (float(v) for v in out[0].split(","))
                return util, mem
        return None

    def run(self) -> None:
        while not self._halt.wait(self.interval):
            try:
                gpu = self._read_gpu()
                if gpu:
                    self.gpu_util.append(gpu[0])
                    self.gpu_mem.append(gpu[1])
            except Exception:  # noqa: BLE001
                pass
            if self._psutil:
                self.cpu.append(self._psutil.cpu_percent(interval=None))

    def finish(self) -> None:
        self._halt.set()
        self.join(timeout=2)


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


# --- 한 조합·한 인원 수 측정 ---
@dataclass
class Row:
    workers: int
    batch: bool
    fp16: bool
    webcam: bool
    examinees: int
    seconds: float = 0.0
    fps_phone: float | None = None
    fps_webcam: float | None = None  # 웹캠 off 면 비움
    fps_min: float | None = None
    lat_p50_ms: float | None = None
    lat_p95_ms: float | None = None
    drop_pct: float | None = None
    gpu_util_pct: float | None = None
    gpu_mem_mb: float | None = None
    cpu_pct: float | None = None
    cpu_cores: int = 0
    threads_req: int = 0  # 요청한 워커당 스레드(0=자동)
    threads: int = 0  # 실제 워커당 스레드
    cpu_eff: int = 0  # 실효 CPU
    cgroup: str = ""  # cgroup 원문
    stage_ms: dict[str, float] = field(default_factory=dict)
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.fps_min is not None and self.fps_min >= PASS_FPS


@dataclass
class Counters:
    sent: Counter = field(default_factory=Counter)  # 채널 -> 보낸 수
    dropped: Counter = field(default_factory=Counter)  # 채널 -> 밀려 버려진 수
    done: Counter = field(default_factory=Counter)  # 채널 -> 분석 끝난 수
    latencies: list[float] = field(default_factory=list)  # 초


async def simulate(detector: Detector, examinees: int, duration: float, warmup: float, photos: tuple[list[bytes], list[bytes]], channels: tuple[str, ...]) -> tuple[Counters, float]:
    """가상 응시자 N명이 `duration` 초(웜업 `warmup` 초 별도) 동안 초당 1장씩 채널마다 보낸다. (집계, 측정 시간 초) 를 돌려준다."""
    phone_photos, webcam_photos = photos
    boxes = Mailboxes()
    arrived: dict[tuple[str, str], float] = {}
    wake = asyncio.Event()
    stats = Counters()
    measuring = False
    t_start = time.monotonic()
    t_measure = t_start + warmup
    t_end = t_measure + duration
    seqs: Counter = Counter()

    async def produce() -> None:
        nonlocal measuring
        k = 0
        while True:
            for i in range(examinees):
                due = t_start + k + i / examinees  # 응시자마다 시작 시각을 나눠 한꺼번에 몰리지 않게
                await asyncio.sleep(max(0.0, due - time.monotonic()))
                now = time.monotonic()
                if now >= t_end:
                    return
                measuring = now >= t_measure
                for channel in channels:
                    key = f"c{i:03d}"
                    seqs[key, channel] += 1
                    seq = seqs[key, channel]
                    pool = phone_photos if channel == PHONE else webcam_photos
                    jpeg = pool[(i + seq) % len(pool)]
                    replaced = boxes.put(ParsedFrame(key, channel, seq, int(now * 1000), jpeg))
                    if measuring:
                        stats.sent[channel] += 1
                        if replaced:
                            stats.dropped[channel] += 1
                    arrived[key, channel] = now
                wake.set()
            k += 1

    async def consume() -> None:
        while True:
            await wake.wait()
            wake.clear()
            frames = boxes.take_all()
            if not frames:
                continue
            batch = [Frame(f.key, f.channel, f.seq, f.recv_ms, f.jpeg, t_start_ms) for f in frames]
            try:
                dets = await detector.run_async(batch)
            except Exception as err:  # noqa: BLE001 - 한 번 실패해도 측정은 계속, 마지막 오류만 남긴다
                stats.errors = f"실패: {err!r}"[:120]  # type: ignore[attr-defined]
                continue
            now = time.monotonic()
            for d in dets:
                queued = arrived.pop((d.key, d.channel), None)
                if queued is not None and queued >= t_measure:
                    stats.done[d.channel] += 1
                    stats.latencies.append(now - queued)

    t_start_ms = int(t_start * 1000)
    producer = asyncio.create_task(produce())
    consumer = asyncio.create_task(consume())
    await producer
    await asyncio.sleep(0.0)
    # 마지막으로 보낸 프레임이 끝나도록 잠깐(최대 5초) 더 기다린다
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and (arrived or boxes._latest):  # noqa: SLF001
        await asyncio.sleep(0.05)
    consumer.cancel()
    try:
        await consumer
    except asyncio.CancelledError:
        pass
    return stats, duration


def summarize_row(combo: Combo, examinees: int, stats: Counters, seconds: float, sampler: Sampler, stage: dict[str, tuple[int, float]], cores: int) -> Row:
    active = (PHONE, WEBCAM) if combo.webcam else (PHONE,)
    fps = {c: stats.done[c] / examinees / seconds for c in active}
    sent = sum(stats.sent[c] for c in active)
    dropped = sum(stats.dropped[c] for c in active)
    lat = [v * 1000 for v in stats.latencies]
    return Row(
        workers=combo.workers, batch=combo.batch, fp16=combo.fp16, webcam=combo.webcam, examinees=examinees,
        seconds=seconds,
        fps_phone=round(fps[PHONE], 3),
        fps_webcam=round(fps[WEBCAM], 3) if combo.webcam else None,
        fps_min=round(min(fps.values()), 3),
        lat_p50_ms=round(percentile(lat, 0.5), 1) if lat else None,
        lat_p95_ms=round(percentile(lat, 0.95), 1) if lat else None,
        drop_pct=round(100 * dropped / sent, 1) if sent else None,
        gpu_util_pct=_r(mean(sampler.gpu_util)),
        gpu_mem_mb=_r(max(sampler.gpu_mem) if sampler.gpu_mem else None),
        cpu_pct=_r(mean(sampler.cpu)),
        cpu_cores=cores,
        stage_ms={k: round(ms / n, 2) for k, (n, ms) in stage.items() if n},
        note=getattr(stats, "errors", ""),
    )


def _r(value: float | None) -> float | None:
    return None if value is None else round(value, 1)


# --- 결과 요약·출력 ---
def max_examinees(rows: Sequence[Row]) -> dict[tuple, int]:
    """조합별로 응시자당 0.9fps 이상을 유지한 최대 인원(없으면 0)."""
    best: dict[tuple, int] = {}
    for r in rows:
        key = (r.workers, r.batch, r.fp16, r.webcam, r.threads_req)
        best.setdefault(key, 0)
        if r.ok:
            best[key] = max(best[key], r.examinees)
    return best


CSV_FIELDS = (
    "workers", "batch", "fp16", "webcam", "examinees", "seconds", "fps_phone", "fps_webcam", "fps_min", "ok",
    "lat_p50_ms", "lat_p95_ms", "drop_pct", "gpu_util_pct", "gpu_mem_mb", "cpu_pct", "cpu_cores",
    "threads", "cpu_eff", "cgroup", *(f"ms_{s}" for s in STAGES), "note",
)


def row_dict(r: Row) -> dict:
    d = asdict(r)
    stage = d.pop("stage_ms")
    d["batch"], d["fp16"] = ("on" if r.batch else "off"), ("on" if r.fp16 else "off")
    d["webcam"] = "server" if r.webcam else "off"
    d["ok"] = "yes" if r.ok else ("skipped" if r.note == "skipped" else "no")
    for s in STAGES:
        d[f"ms_{s}"] = stage.get(s)
    return {k: ("" if d.get(k) is None else d[k]) for k in CSV_FIELDS}


def write_csv(rows: Sequence[Row], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:  # 엑셀에서 한글이 안 깨지게
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(row_dict(r))


def _cell(v) -> str:
    return "-" if v is None or v == "" else str(v)


def render_markdown(rows: Sequence[Row], env_note: str) -> str:
    best = max_examinees(rows)
    tested = sorted({r.examinees for r in rows if r.note != "skipped"})
    lines = [
        "# 용량 측정 결과",
        "",
        f"측정 환경: {env_note}",
        "",
        f"## 요약: 조합별로 응시자당 {PASS_FPS}fps 이상을 유지한 최대 인원",
        "",
        f"(응시자당 fps = 분석이 끝난 프레임 수 / 응시자 수 / 초. 휴대폰·웹캠 중 낮은 쪽 기준. 측정한 인원: {', '.join(map(str, tested))}명)",
        "",
        "| workers | 스레드/워커 | 배치 | fp16 | 웹캠 | 최대 인원 |",
        "|---|---|---|---|---|---|",
    ]
    actual = {(r.workers, r.batch, r.fp16, r.webcam, r.threads_req): r.threads for r in rows if r.threads}
    for key, n in best.items():
        w, b, f, c, t = key
        lines.append(f"| {w} | {actual.get(key, t) or 'auto'} | {'on' if b else 'off'} | {'on' if f else 'off'} | {'server' if c else 'off'} | {n if n else f'{tested[0]}명 미만' if tested else '-'}{'명' if n else ''} |")
    lines += [
        "",
        "## 상세",
        "",
        "| workers | 스레드 | 배치 | fp16 | 웹캠 | 인원 | fps 휴대폰 | fps 웹캠 | 통과 | 지연 p50 ms | p95 ms | 버림 % | GPU % | GPU MB | CPU % | 코어 | 실효CPU | "
        + " | ".join(f"{s} ms" for s in STAGES) + " | 비고 |",
        "|" + "---|" * (17 + len(STAGES) + 1),
    ]
    for r in rows:
        d = row_dict(r)
        cells = [d["workers"], d["threads"], d["batch"], d["fp16"], d["webcam"], d["examinees"], d["fps_phone"], d["fps_webcam"], d["ok"],
                 d["lat_p50_ms"], d["lat_p95_ms"], d["drop_pct"], d["gpu_util_pct"], d["gpu_mem_mb"], d["cpu_pct"], d["cpu_cores"], d["cpu_eff"],
                 *(d[f"ms_{s}"] for s in STAGES), d["note"]]
        lines.append("| " + " | ".join(_cell(c) for c in cells) + " |")
    lines += [
        "",
        "읽는 법: 통과=yes 는 휴대폰·웹캠 모두 응시자당 0.9fps 이상. 버림 %는 큐 길이 1에서 밀려 버려진 프레임 비율, 단계 ms 는 프레임 한 장당 평균(배치는 묶음 시간을 장수로 나눈 값, 워커가 여러 개면 워커 합산 평균).",
        "웹캠 off 인 행은 웹캠 프레임을 분석하지 않는다(얼굴·포즈 생략). 단계 ms 가 빈 칸이면 그 단계가 없거나(가짜 모델) 돌지 않은 것이다.",
    ]
    return "\n".join(lines) + "\n"


def environment_note(detector_kind: str, gpu_name: str, photo_note: str) -> str:
    gpu = gpu_name or "GPU 정보 없음"
    return f"{platform.platform()}, Python {platform.python_version()}, 코어 {os.cpu_count()} / {cpu_info().describe()}, {gpu}, detector={detector_kind}, {photo_note}"


# --- 실행 ---
def run_combo(combo: Combo, examinee_counts: Sequence[int], args: argparse.Namespace, photos, stop_early: bool = True) -> list[Row]:
    rules = None
    if args.detector == "real":
        from pair_vision.rules.config import load_rules

        rules = load_rules(args.rules).models
    # 스레드: 조합이 지정한 값 > (--thread-cap 이 있으면 min(상한, 실효 CPU ÷ workers)) > 자동(실효 CPU ÷ workers)
    cap = getattr(args, "thread_cap", 0)
    requested = combo.threads or (min(cap, default_threads(combo.workers)) if cap else 0)
    detector = create_detector(
        args.detector, args.scenario, models_dir=args.models_dir, model_rules=rules,
        lock_path=args.models_lock, verify=not args.no_verify, workers=combo.workers, options=combo.options(),
        worker_threads=requested,
    )
    t0 = time.monotonic()
    detector.load()
    info = cpu_info()
    threads = getattr(detector, "threads", 0) or resolve_threads(requested, combo.workers)
    print(f"[{combo.label()}] 모델 적재 {time.monotonic() - t0:.1f}초, 워커당 스레드 {threads}, {info.describe()}", flush=True)
    rows: list[Row] = []
    failed = False
    try:
        for n in sorted(examinee_counts):
            if failed and stop_early:
                rows.append(Row(combo.workers, combo.batch, combo.fp16, combo.webcam, n, note="skipped", threads_req=combo.threads,
                                threads=threads, cpu_eff=info.effective, cgroup=info.cgroup_raw))
                continue
            sampler = Sampler()
            detector.stage_stats(reset=True)
            sampler.start()
            channels = (PHONE, WEBCAM) if combo.webcam else (PHONE, WEBCAM)  # 웹캠 off 여도 프레임은 도착한다(분석만 안 함)
            stats, seconds = asyncio.run(simulate(detector, n, args.duration, args.warmup, photos, channels))
            sampler.finish()
            row = summarize_row(combo, n, stats, seconds, sampler, detector.stage_stats(reset=True), os.cpu_count() or 0)
            row.threads_req, row.threads, row.cpu_eff, row.cgroup = combo.threads, threads, info.effective, info.cgroup_raw
            rows.append(row)
            print(
                f"  {n:>3}명: fps 휴대폰 {_cell(row.fps_phone)} 웹캠 {_cell(row.fps_webcam)} | p50 {_cell(row.lat_p50_ms)}ms "
                f"p95 {_cell(row.lat_p95_ms)}ms | 버림 {_cell(row.drop_pct)}% | GPU {_cell(row.gpu_util_pct)}% | {'통과' if row.ok else '미달'}",
                flush=True,
            )
            failed = failed or not row.ok
    finally:
        detector.close()
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--detector", default="fake", choices=["fake", "real"])
    p.add_argument("--scenario", default="normal", help="가짜 모델 시나리오")
    p.add_argument("--photos", type=Path, default=None, help="사진 폴더(없으면 합성 사진)")
    p.add_argument("--webcam-photos", type=Path, default=None, help="웹캠용 사진 폴더(기본: --photos 와 같음)")
    p.add_argument("--workers", default=DEFAULT_WORKERS)
    p.add_argument("--worker-threads", default="0", help="워커당 스레드 수 목록(0=자동: 실효 CPU ÷ workers)")
    p.add_argument("--thread-cap", type=int, default=0, help="워커당 스레드 상한(0=없음). 자동값(실효 CPU ÷ workers)이 이보다 크면 이 값으로")
    p.add_argument("--batch", default="off,on", help="GPU 묶음 추론(VISION_BATCH_GPU): off,on")
    p.add_argument("--fp16", default="off,on", help="VISION_FP16: off,on")
    p.add_argument("--webcam", default="server,off", help="웹캠 서버 분석(VISION_WEBCAM): server,off")
    p.add_argument("--examinees", default=DEFAULT_EXAMINEES, help="가상 응시자 수 목록")
    p.add_argument("--duration", type=float, default=20.0, help="측정 시간(초)")
    p.add_argument("--warmup", type=float, default=5.0, help="웜업(초, 집계 제외)")
    p.add_argument("--full", action="store_true", help="한 인원 수에서 미달해도 더 큰 인원 수까지 계속 잰다(기본은 건너뜀)")
    p.add_argument("--phone-edge", type=int, default=1280)
    p.add_argument("--phone-quality", type=int, default=85)
    p.add_argument("--webcam-edge", type=int, default=400, help="계약: 웹캠 긴 변 400")
    p.add_argument("--webcam-quality", type=int, default=55, help="계약: 품질 0.55")
    p.add_argument("--models-dir", default=os.environ.get("VISION_MODELS_DIR", "models"))
    p.add_argument("--models-lock", default=os.environ.get("VISION_MODELS_LOCK", "configs/models.lock"))
    p.add_argument("--rules", default=os.environ.get("VISION_RULES_PATH", "configs/rules.default.yaml"))
    p.add_argument("--no-verify", action="store_true", help="모델 sha256 검증 건너뜀")
    p.add_argument("--out", type=Path, default=Path("capacity"), help="결과 파일 접두어(.csv .md 가 붙는다)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    combos = build_combos(
        parse_ints(args.workers),
        parse_choices(args.batch, ("off", "on")),
        parse_choices(args.fp16, ("off", "on")),
        parse_choices(args.webcam, ("off", "server")),
        parse_threads(args.worker_threads),
    )
    counts = parse_ints(args.examinees)
    phone, webcam, note = make_photo_set(args.photos, args.phone_edge, args.webcam_edge, args.phone_quality, args.webcam_quality)
    if args.webcam_photos:
        _, webcam, _ = make_photo_set(args.webcam_photos, args.phone_edge, args.webcam_edge, args.phone_quality, args.webcam_quality)
    gpu_probe = Sampler()
    print(f"조합 {len(combos)}개 x 인원 {counts}, 측정 {args.duration}초(+웜업 {args.warmup}초) | {note}", flush=True)
    all_rows: list[Row] = []
    for combo in combos:
        try:
            all_rows += run_combo(combo, counts, args, (phone, webcam), stop_early=not args.full)
        except Exception as err:  # noqa: BLE001 - 한 조합이 실패해도 나머지를 잰다(예: 워커 8개에서 GPU 메모리 부족)
            print(f"[{combo.label()}] 실패: {err!r}", flush=True)
            all_rows += [Row(combo.workers, combo.batch, combo.fp16, combo.webcam, n, note=f"실패: {err!r}"[:120], threads_req=combo.threads)
                         for n in counts]
        # 조합마다 중간 저장(Pod 가 중간에 끊겨도 결과가 남게)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_csv(all_rows, args.out.with_suffix(".csv"))
        args.out.with_suffix(".md").write_text(render_markdown(all_rows, environment_note(args.detector, gpu_probe.gpu_name, note)), encoding="utf-8")
    print(f"\n끝: {args.out.with_suffix('.csv')}, {args.out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
