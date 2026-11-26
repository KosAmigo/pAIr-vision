"""모델 출력 후처리(순수 파이썬 - torch·mediapipe 없이 시험된다)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from pair_vision.models.base import Box, HandDet, ObjectDet

# D-FINE(COCO) 라벨 -> 서버 라벨. 노트는 COCO book.
COCO_LABELS = {"keyboard": "keyboard", "mouse": "mouse", "book": "note"}
# 방 스캔(POST /v1/scan): COCO 라벨 -> 스캔 라벨. person·tv·laptop 기준값은 고정, phone·book 은 규칙 파일 값(휴대폰·노트).
SCAN_COCO_LABELS = {"person": "person", "cell phone": "phone", "tv": "tv", "laptop": "laptop", "book": "book"}
SCAN_FIXED_THRESHOLDS = {"person": 0.5, "tv": 0.5, "laptop": 0.5}
# OWLv2 문장 -> 서버 라벨
OWL_PROMPTS = {
    "a tablet": "tablet",
    "a smartwatch on a wrist": "smartwatch",
    "a smartphone": "phone",
}


def box_in_image(box: Sequence[float], width: int, height: int) -> Box:
    """픽셀 박스를 사진 크기로 나눠 0~1 로 바꾼다(사진 밖은 잘라낸다).

    OWLv2 는 긴 변 기준 정사각형 패딩 평면의 좌표를 내고 사진은 그 왼쪽 위에 놓이므로
    사진의 가로·세로로 나누면 곧바로 사진 기준 좌표다.
    """
    x0, y0, x1, y1 = (
        min(1.0, max(0.0, v))
        for v in (box[0] / width, box[1] / height, box[2] / width, box[3] / height)
    )
    return Box(x0, y0, x1, y1)


def merge_close_hands(hands: Iterable[HandDet], distance: float) -> list[HandDet]:
    """가까운 손 중복을 하나로 합친다(점수 높은 쪽 유지). 결과는 점수 내림차순."""
    kept: list[HandDet] = []
    for hand in sorted(hands, key=lambda h: h.score, reverse=True):
        if all(math.hypot(hand.x - k.x, hand.y - k.y) >= distance for k in kept):
            kept.append(hand)
    return kept


def best_per_label(objects: Iterable[ObjectDet]) -> dict[str, ObjectDet]:
    best: dict[str, ObjectDet] = {}
    for obj in objects:
        if obj.label not in best or obj.score > best[obj.label].score:
            best[obj.label] = obj
    return best
