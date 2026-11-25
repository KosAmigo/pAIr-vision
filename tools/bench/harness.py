"""모델 비교 측정 틀(순수 파이썬 — numpy·torch 없이 import 된다).

같은 사진 폴더를 여러 `BenchModel` 에 돌려 **속도(ms/장)** 와 **검출 요약**을 낸다.
정확도(정답 대비)는 계산하지 않는다 — 정답 라벨이 있는 사진이 생긴 뒤 tools/eval 에서 한다.
"""

from __future__ import annotations

import statistics
import time
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pair_vision.models.base import Box, Detection, Detector, Frame, HandDet

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
KEYBOARD_LABELS = ("keyboard",)
MOUSE_LABELS = ("mouse",)


@dataclass(frozen=True)
class ObjectDet:
    """모델이 낸 물체 하나. 좌표는 프레임 비율(0~1)."""

    label: str
    score: float
    box: Box


@dataclass(frozen=True)
class RawResult:
    objects: tuple[ObjectDet, ...] = ()
    hands: tuple[HandDet, ...] = ()


class BenchModel(Detector):
    """비교 대상 모델의 공통 틀. `run(frames) -> detections` 는 Detector 계약 그대로 만족한다.

    하위 클래스는 `infer(jpeg)` 만 채운다. 키보드·마우스·손만 `Detection` 으로 옮겨지고,
    휴대폰·태블릿 등 나머지 물체는 `infer` 결과(RawResult)로 측정 보고서에 남는다.
    """

    @abstractmethod
    def infer(self, jpeg: bytes) -> RawResult: ...

    def run(self, frames: Sequence[Frame]) -> list[Detection]:
        return [to_detection(frame, self.infer(frame.jpeg)) for frame in frames]


def best_box(objects: Iterable[ObjectDet], labels: Sequence[str]) -> Box | None:
    candidates = [o for o in objects if o.label in labels]
    return max(candidates, key=lambda o: o.score).box if candidates else None


def to_detection(frame: Frame, raw: RawResult) -> Detection:
    return Detection(
        key=frame.key,
        channel=frame.channel,
        seq=frame.seq,
        recv_ms=frame.recv_ms,
        keyboard=best_box(raw.objects, KEYBOARD_LABELS),
        mouse=best_box(raw.objects, MOUSE_LABELS),
        hands=raw.hands,
    )


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"사진 폴더가 없습니다: {folder}")
    images = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"사진(.jpg/.jpeg/.png)이 없습니다: {folder}")
    return images


def percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


@dataclass(frozen=True)
class TimingStats:
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float

    @classmethod
    def from_samples(cls, samples_ms: Sequence[float]) -> TimingStats:
        if not samples_ms:
            return cls(0, 0.0, 0.0, 0.0, 0.0)
        return cls(
            count=len(samples_ms),
            mean_ms=statistics.fmean(samples_ms),
            p50_ms=percentile(samples_ms, 0.5),
            p95_ms=percentile(samples_ms, 0.95),
            max_ms=max(samples_ms),
        )


@dataclass
class ModelReport:
    model: str
    gpu: bool
    load_ms: float
    timing: TimingStats
    # 라벨 -> (검출한 사진 수, 점수 평균)
    label_summary: dict[str, tuple[int, float]] = field(default_factory=dict)
    keyboard_found: int = 0
    mouse_found: int = 0
    hands_mean: float = 0.0
    images: int = 0
    per_image: dict[str, list[dict]] = field(default_factory=dict)


def summarize(raws: dict[str, RawResult]) -> dict[str, tuple[int, float]]:
    seen: dict[str, set[str]] = defaultdict(set)
    scores: dict[str, list[float]] = defaultdict(list)
    for name, raw in raws.items():
        for obj in raw.objects:
            seen[obj.label].add(name)
            scores[obj.label].append(obj.score)
    return {label: (len(seen[label]), statistics.fmean(scores[label])) for label in sorted(seen)}


def measure(
    model: BenchModel,
    images: Sequence[Path],
    *,
    warmup: int = 2,
    repeat: int = 1,
    clock: Callable[[], float] = time.perf_counter,
    read: Callable[[Path], bytes] = Path.read_bytes,
) -> ModelReport:
    """`model` 을 적재하고 사진마다 `repeat` 번 돌려 ms/장 을 잰다.

    JPEG 읽기(디스크)는 시간에 넣지 않는다 — 서버에서는 이미 메모리에 있는 바이트가 들어온다.
    JPEG 디코드는 모델 안에서 일어나므로 시간에 들어간다."""
    t0 = clock()
    model.load()
    load_ms = (clock() - t0) * 1000

    blobs = {path.name: read(path) for path in images}
    for _ in range(max(0, warmup)):
        model.infer(next(iter(blobs.values())))

    samples: list[float] = []
    raws: dict[str, RawResult] = {}
    for name, blob in blobs.items():
        for _ in range(max(1, repeat)):
            start = clock()
            raw = model.infer(blob)
            samples.append((clock() - start) * 1000)
        raws[name] = raw

    return ModelReport(
        model=model.name,
        gpu=model.gpu,
        load_ms=load_ms,
        timing=TimingStats.from_samples(samples),
        label_summary=summarize(raws),
        keyboard_found=sum(1 for r in raws.values() if best_box(r.objects, KEYBOARD_LABELS)),
        mouse_found=sum(1 for r in raws.values() if best_box(r.objects, MOUSE_LABELS)),
        hands_mean=statistics.fmean(len(r.hands) for r in raws.values()),
        images=len(raws),
        per_image={
            name: [
                {
                    "label": o.label,
                    "score": round(o.score, 3),
                    "box": [round(v, 3) for v in (o.box.x0, o.box.y0, o.box.x1, o.box.y1)],
                }
                for o in raw.objects
            ]
            for name, raw in raws.items()
        },
    )


def render_markdown(reports: Sequence[ModelReport], device_note: str) -> str:
    lines = [
        f"측정 환경: {device_note}",
        "",
        "| 모델 | GPU | 적재 ms | ms/장 평균 | p50 | p95 | 사진 수 | 키보드 찾은 사진 | 마우스 찾은 사진 | 손 평균 개수 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        t = r.timing
        lines.append(
            f"| {r.model} | {'예' if r.gpu else '아니오'} | {r.load_ms:.0f} | {t.mean_ms:.1f} | "
            f"{t.p50_ms:.1f} | {t.p95_ms:.1f} | {r.images} | {r.keyboard_found} | "
            f"{r.mouse_found} | {r.hands_mean:.2f} |"
        )
    lines += ["", "라벨별 검출(검출한 사진 수 / 평균 점수) — 정답 대비 정확도가 아니다:"]
    for r in reports:
        items = ", ".join(f"{k} {n}장/{s:.2f}" for k, (n, s) in r.label_summary.items()) or "없음"
        lines.append(f"- {r.model}: {items}")
    return "\n".join(lines)
