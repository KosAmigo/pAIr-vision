"""Detection -> 규칙 엔진 입력(HandFrame) 변환. 영역 = 키보드 영역 ∪ 마우스 영역."""

from __future__ import annotations

from collections.abc import Iterable

from pair_vision.models.base import Box, Detection
from pair_vision.rules.hands import HandFrame

MAX_HANDS = 2


def to_hand_frame(det: Detection, area: Iterable[Box], margin: float) -> HandFrame:
    boxes = [b.expanded(margin) for b in area]
    hands = det.hands[:MAX_HANDS]
    inside = sum(1 for h in hands if any(b.contains(h.x, h.y) for b in boxes))
    return HandFrame(recv_ms=det.recv_ms, inside=inside, outside=len(hands) - inside)
