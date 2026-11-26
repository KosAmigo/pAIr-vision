"""웹캠 규칙 판정 (프론트 `core/pcRules.ts` 포팅): 보정 자세 대비 차이로 조건을 만든다.

조건 키 = 규칙 라벨: no_face multi_face head_turn head_down gaze_away torso_shift torso_turn.
조건이 일정 시간 이어지면 이벤트로 만드는 부분은 `rules/objects.py` 의 ObjectRuleEngine 을 재사용한다.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from pair_vision.models.base import Shoulders, WebcamFeatures

# 규칙 라벨 -> 이벤트 유형 (모두 source=PC_CAM)
WEBCAM_EVENT_TYPES = {
    "no_face": "NO_FACE",
    "multi_face": "MULTI_FACE",
    "head_turn": "HEAD_TURN",
    "head_down": "HEAD_DOWN",
    "gaze_away": "GAZE_AWAY",
    "torso_shift": "TORSO_SHIFT",
    "torso_turn": "TORSO_TURN",
}


@dataclass(frozen=True)
class WebcamThresholds:
    """프론트 PC_THRESHOLDS 와 같은 값."""

    yaw_deg: float = 25
    pitch_up_deg: float = 20
    pitch_down_deg: float = 25
    gaze_x: float = 0.45
    gaze_down: float = 0.45
    blink_ignore: float = 0.5
    torso_shift: float = 0.12
    torso_width_min: float = 0.72
    torso_width_max: float = 1.35


@dataclass(frozen=True)
class WebcamBaseline:
    yaw: float
    pitch: float
    gaze_x: float
    gaze_y: float
    shoulders: Shoulders | None


def angle_diff(a: float, b: float) -> float:
    """각도 차이를 -180~180 으로."""
    d = a - b
    while d > 180:
        d -= 360
    while d < -180:
        d += 360
    return d


def webcam_conditions(f: WebcamFeatures, b: WebcamBaseline, t: WebcamThresholds) -> dict[str, bool]:
    cond = {"no_face": f.face_count == 0, "multi_face": f.face_count > 1}
    if f.face_count >= 1 and f.yaw is not None and f.pitch is not None:
        d_yaw = angle_diff(f.yaw, b.yaw)
        d_pitch = angle_diff(f.pitch, b.pitch)
        cond["head_turn"] = abs(d_yaw) > t.yaw_deg or d_pitch < -t.pitch_up_deg
        cond["head_down"] = d_pitch > t.pitch_down_deg
    eyes_open = (f.blink if f.blink is not None else 0.0) < t.blink_ignore
    if f.face_count >= 1 and eyes_open and f.gaze_x is not None and f.gaze_y is not None:
        cond["gaze_away"] = abs(f.gaze_x - b.gaze_x) > t.gaze_x or f.gaze_y - b.gaze_y > t.gaze_down
    if b.shoulders is not None:
        if f.shoulders is None:
            cond["torso_shift"] = True
        else:
            shift = math.hypot(f.shoulders.cx - b.shoulders.cx, f.shoulders.cy - b.shoulders.cy)
            ratio = f.shoulders.width / b.shoulders.width
            cond["torso_shift"] = shift > t.torso_shift
            cond["torso_turn"] = ratio < t.torso_width_min or ratio > t.torso_width_max
    return cond


def build_baseline(
    samples: Sequence[WebcamFeatures], *, min_samples: int, min_face_ratio: float = 0.6
) -> WebcamBaseline | None:
    """보정 구간 샘플의 중앙값으로 기준 자세를 만든다. 정면 얼굴(1개)이 충분히 안 잡히면 None.

    프론트는 최소 5개를 요구하지만 서버는 초당 1장이라 3초 보정에 4장쯤 모이므로 `min_samples` 를 설정으로 받는다.
    """
    faces = [s for s in samples if s.face_count == 1 and s.yaw is not None and s.pitch is not None]
    if len(faces) < max(min_samples, len(samples) * min_face_ratio):
        return None
    shoulders = [s.shoulders for s in samples if s.shoulders is not None]
    return WebcamBaseline(
        yaw=statistics.median(s.yaw for s in faces),
        pitch=statistics.median(s.pitch for s in faces),
        gaze_x=statistics.median(s.gaze_x or 0.0 for s in faces),
        gaze_y=statistics.median(s.gaze_y or 0.0 for s in faces),
        shoulders=(
            Shoulders(
                cx=statistics.median(s.cx for s in shoulders),
                cy=statistics.median(s.cy for s in shoulders),
                width=statistics.median(s.width for s in shoulders),
            )
            if len(shoulders) >= len(samples) * 0.5
            else None
        ),
    )
