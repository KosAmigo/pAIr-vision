"""MediaPipe 얼굴·포즈 결과 -> WebcamFeatures (프론트 `vision/features.ts` 포팅). 순수 파이썬: mediapipe 없이 시험된다.

입력은 일반 파이썬 값이다(행렬은 4x4 중첩 시퀀스, 블렌드셰이프는 {이름: 점수}, 랜드마크는 x·y·visibility 속성).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Protocol

from pair_vision.models.base import Shoulders, WebcamFeatures

RAD = 180 / math.pi
VISIBLE = 0.5
L_SHOULDER = 11
R_SHOULDER = 12


class Landmark(Protocol):
    x: float
    y: float


def head_pose_from_matrix(m: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    """얼굴 변환 행렬 4x4(m[행][열])에서 (yaw, pitch, roll) 도. pitch 는 +가 고개 숙임.

    프론트는 column-major 평면 배열 data 에서 r(row, col) = data[col*4 + row] 를 쓴다. 같은 행렬을
    m[row][col] 로 읽으면 같은 값이다.
    """
    yaw = math.asin(max(-1.0, min(1.0, -m[2][0]))) * RAD
    pitch = math.atan2(m[2][1], m[2][2]) * RAD
    roll = math.atan2(m[1][0], m[0][0]) * RAD
    return yaw, pitch, roll


def gaze_from_blendshapes(scores: Mapping[str, float]) -> tuple[float, float, float]:
    """(gaze_x, gaze_y, blink). Left/Right 는 응시자 기준: 오른쪽을 보면 왼눈은 In, 오른눈은 Out."""

    def v(key: str) -> float:
        return scores.get(key, 0.0)

    gaze_x = (v("eyeLookInLeft") - v("eyeLookOutLeft") + (v("eyeLookOutRight") - v("eyeLookInRight"))) / 2
    gaze_y = (v("eyeLookDownLeft") - v("eyeLookUpLeft") + (v("eyeLookDownRight") - v("eyeLookUpRight"))) / 2
    blink = (v("eyeBlinkLeft") + v("eyeBlinkRight")) / 2
    return gaze_x, gaze_y, blink


def _visible(lm: object) -> bool:
    x, y = getattr(lm, "x", -1.0), getattr(lm, "y", -1.0)
    visibility = getattr(lm, "visibility", None)
    return (visibility is None or visibility >= VISIBLE) and 0 <= x <= 1 and 0 <= y <= 1


def shoulders_from_pose(landmarks: Sequence[Landmark] | None) -> Shoulders | None:
    if not landmarks or len(landmarks) <= R_SHOULDER:
        return None
    left, right = landmarks[L_SHOULDER], landmarks[R_SHOULDER]
    if not (_visible(left) and _visible(right)):
        return None
    return Shoulders(
        cx=(left.x + right.x) / 2,
        cy=(left.y + right.y) / 2,
        width=math.hypot(left.x - right.x, left.y - right.y),
    )


def webcam_features(
    face_count: int,
    matrix: Sequence[Sequence[float]] | None,
    blendshapes: Mapping[str, float] | None,
    pose_landmarks: Sequence[Landmark] | None,
) -> WebcamFeatures:
    yaw = pitch = gaze_x = gaze_y = blink = None
    if face_count >= 1:
        if matrix is not None:
            yaw, pitch, _ = head_pose_from_matrix(matrix)
        if blendshapes is not None:
            gaze_x, gaze_y, blink = gaze_from_blendshapes(blendshapes)
    return WebcamFeatures(
        face_count=face_count, yaw=yaw, pitch=pitch, gaze_x=gaze_x, gaze_y=gaze_y, blink=blink,
        shoulders=shoulders_from_pose(pose_landmarks),
    )
