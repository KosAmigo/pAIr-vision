"""모델 래퍼 인터페이스 `Detector.run(frames) -> detections` (설계 §15-3 원칙)."""

from __future__ import annotations

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


class Detector(ABC):
    """모델 교체 지점. 규칙 엔진은 이 출력(Detection)만 본다."""

    name: str = "abstract"
    gpu: bool = False

    def load(self) -> None:  # noqa: B027 - 가짜 구현은 적재할 것이 없다
        """모델 적재. 끝나면 /readyz 의 modelsLoaded 가 참이 된다."""

    @abstractmethod
    def run(self, frames: Sequence[Frame]) -> list[Detection]:
        """프레임 묶음을 같은 순서의 Detection 목록으로 바꾼다."""
