"""모델 래퍼 인터페이스 `Detector.run(frames) -> detections` (설계 §15-3 원칙)."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

WEBCAM = "webcam"
PHONE = "phone"


@dataclass(frozen=True)
class Frame:
    """추론에 넘기는 프레임. 사진 바이트는 메모리에만 있고 디스크·로그에 쓰지 않는다."""

    key: str
    channel: str
    seq: int
    recv_ms: int
    jpeg: bytes = b""
    # 가짜 시나리오 재생기가 "세션 시작 후 몇 초"를 알 수 있게 파이프라인이 채운다.
    origin_ms: int = 0


@dataclass(frozen=True)
class Box:
    """프레임 비율(0~1) 좌표."""

    x0: float
    y0: float
    x1: float
    y1: float

    def expanded(self, margin: float) -> Box:
        return Box(self.x0 - margin, self.y0 - margin, self.x1 + margin, self.y1 + margin)

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


@dataclass(frozen=True)
class HandDet:
    side: str  # "left" | "right" | "unknown"
    x: float
    y: float
    score: float = 1.0  # 손 검출 점수(0~1). 규칙은 점수 상위 2개만 쓴다.


@dataclass(frozen=True)
class ObjectDet:
    """모델이 낸 물체 하나. 좌표는 프레임 비율(0~1). label 은 짧은 이름(keyboard·mouse·note·tablet·smartwatch·phone)."""

    label: str
    score: float
    box: Box


@dataclass(frozen=True)
class Shoulders:
    """어깨 중심(cx, cy)과 너비. 프레임 비율(0~1)."""

    cx: float
    cy: float
    width: float


@dataclass(frozen=True)
class WebcamFeatures:
    """웹캠 한 프레임의 얼굴·몸통 특징값 (프론트 pcRules.ts 의 PcFeatures 와 같은 뜻)."""

    face_count: int
    yaw: float | None = None  # 고개 좌우 회전(도)
    pitch: float | None = None  # 고개 상하(도), +는 숙임
    gaze_x: float | None = None  # +는 응시자 기준 오른쪽(-1~1)
    gaze_y: float | None = None  # +는 아래(-1~1)
    blink: float | None = None  # 눈 감김(0~1)
    shoulders: Shoulders | None = None


@dataclass(frozen=True)
class Detection:
    key: str
    channel: str
    seq: int
    recv_ms: int
    keyboard: Box | None = None
    mouse: Box | None = None
    hands: tuple[HandDet, ...] = ()
    desk_visible: bool = True
    # 키보드·마우스 외에 이번 프레임에서 기준값을 넘긴 물체(note·tablet·smartwatch·phone).
    # OWLv2 물체는 그 프레임에서 OWLv2 를 돌렸을 때만 들어 있다(몇 초 주기).
    objects: tuple[ObjectDet, ...] = ()
    # 중복 병합 뒤 검출된 손 전체 수(hands 는 점수 상위 2개만 담는다). None 이면 len(hands).
    hand_count: int | None = None
    webcam: WebcamFeatures | None = None  # 웹캠 채널 프레임의 얼굴·포즈 특징(휴대폰 프레임이면 None)


class Detector(ABC):
    """모델 교체 지점. 규칙 엔진은 이 출력(Detection)만 본다."""

    name: str = "abstract"
    gpu: bool = False
    threads: int = 0  # 이 Detector(워커 하나)가 쓰는 스레드 수(측정 기록용)

    def load(self) -> None:  # noqa: B027 - 가짜 구현은 적재할 것이 없다
        """모델 적재. 끝나면 /readyz 의 modelsLoaded 가 참이 된다."""

    def close(self) -> None:  # noqa: B027 - 서버 종료 때 자원(워커 프로세스 등)을 정리한다
        """서버 종료 알림."""

    async def run_async(self, frames: Sequence[Frame]) -> list[Detection]:
        """파이프라인이 부르는 실행. 기본은 스레드 하나에서 run 을 돌린다(프로세스 풀은 이것을 병렬로 바꾼다)."""
        return await asyncio.to_thread(self.run, frames)

    def forget(self, key: str) -> None:  # noqa: B027 - 세션이 끝났을 때 세션별 상태를 비운다
        """세션 종료 알림. 세션별 상태(예: OWLv2 주기)를 가진 구현이 비운다."""

    @abstractmethod
    def run(self, frames: Sequence[Frame]) -> list[Detection]:
        """프레임 묶음을 같은 순서의 Detection 목록으로 바꾼다."""

    def stage_stats(self, reset: bool = False) -> dict[str, tuple[int, float]]:  # noqa: B027
        """단계별 누적 시간 {단계: (프레임 수, 총 ms)}. 측정 도구(tools/bench/capacity.py)가 읽는다. 기본은 없음."""
        return {}

    def scan(self, jpegs: Sequence[bytes]) -> list[list[ObjectDet]]:
        """방 스캔: 사진마다 person·phone·tv·laptop·book·tablet 물체 목록(좌표는 0~1 비율). 사진을 못 풀면 ScanDecodeError."""
        raise NotImplementedError(f"{self.name} 는 방 스캔을 지원하지 않습니다")

    async def scan_async(self, jpegs: Sequence[bytes]) -> list[list[ObjectDet]]:
        return await asyncio.to_thread(self.scan, jpegs)


class ScanDecodeError(ValueError):
    """방 스캔 사진을 읽을 수 없다(깨진 JPEG 등). API 는 400 BAD_SCAN 으로 바꾼다."""
