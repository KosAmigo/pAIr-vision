"""RTMW(COCO-WholeBody 133점) 결과 -> 손·웹캠 특징값. numpy·cv2 만 쓰는 순수 계산이라 모델 없이 시험된다.

COCO-WholeBody 키포인트 번호: 몸 0~16(COCO 17점, 어깨는 5·6) · 발 17~22 · 얼굴 23~90(68점, iBUG 순서) · 왼손 91~111 · 오른손 112~132.
(왼/오른은 사람 기준. 손 21점의 0 은 손목, 9 는 가운뎃손가락 시작 관절.)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from pair_vision.models.base import HandDet, Shoulders, WebcamFeatures

LEFT_HAND = 91
RIGHT_HAND = 112
FACE_START, FACE_END = 23, 91
L_SHOULDER, R_SHOULDER = 5, 6
VISIBLE = 0.5

# 머리 각도 계산에 쓰는 얼굴 6점(iBUG 68점 번호): 코끝 30, 턱 8, 눈 바깥 끝 36·45, 입 끝 48·54.
FACE_PNP_IDX = (30, 8, 36, 45, 48, 54)
# 같은 6점의 일반적인 3D 얼굴 모델(카메라 좌표: x 오른쪽, y 아래, z 앞). 사람마다 다른 실제 얼굴과는 차이가 있다.
FACE_MODEL_3D = np.array(
    [(0, 0, 0), (0, 330, 65), (-225, -170, 135), (225, -170, 135), (-150, 150, 125), (150, 150, 125)], dtype=np.float64
)


def hands_from_keypoints(
    kps: np.ndarray, scores: np.ndarray, width: int, height: int, threshold: float
) -> list[HandDet]:
    """손 21점의 평균 점수가 기준값 이상인 손만 낸다. 위치 = 손목(0)과 가운뎃손가락 시작(9)의 중점(프레임 비율). 점수 내림차순."""
    hands: list[HandDet] = []
    for side, start in (("left", LEFT_HAND), ("right", RIGHT_HAND)):
        score = float(np.mean(scores[start : start + 21]))
        if score < threshold:
            continue
        wrist, mcp = kps[start], kps[start + 9]
        x = float((wrist[0] + mcp[0]) / 2) / width
        y = float((wrist[1] + mcp[1]) / 2) / height
        hands.append(HandDet(side, x, y, score))
    return sorted(hands, key=lambda h: h.score, reverse=True)


def face_score(scores: np.ndarray) -> float:
    return float(np.mean(scores[FACE_START:FACE_END]))


def head_pose_raw(face68: np.ndarray, width: int, height: int) -> tuple[float, float, float] | None:
    """얼굴 6점 PnP 로 (yaw, pitch, roll) 도. pitch 는 +가 고개 숙임. 실패하면 None. 크기 보정 전의 값."""
    import cv2

    pts = np.array([face68[i] for i in FACE_PNP_IDX], dtype=np.float64)
    cam = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(FACE_MODEL_3D, pts, cam, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rot, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
    pitch = math.degrees(math.atan2(rot[2, 1], rot[2, 2]))
    yaw = math.degrees(math.atan2(-rot[2, 0], sy))
    roll = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
    return yaw, pitch, roll


def head_pose(
    face68: np.ndarray, width: int, height: int, yaw_scale: float, pitch_scale: float
) -> tuple[float, float] | None:
    """MediaPipe 와 같은 부호(yaw: 응시자 오른쪽으로 돌리면 +, pitch: 숙이면 +)로 맞춘 머리 각도.

    6점 일반 모델의 yaw 는 실제보다 크게 나오고 흔들려서 `yaw_scale` 로 줄인다(본인 사진 10장에서 구한 값, 다른 사람으로 재확인 필요).
    """
    raw = head_pose_raw(face68, width, height)
    if raw is None:
        return None
    yaw, pitch, _ = raw
    return -yaw * yaw_scale, pitch * pitch_scale


def shoulders_from_keypoints(kps: np.ndarray, scores: np.ndarray, width: int, height: int) -> Shoulders | None:
    left, right = kps[L_SHOULDER], kps[R_SHOULDER]
    if scores[L_SHOULDER] < VISIBLE or scores[R_SHOULDER] < VISIBLE:
        return None
    lx, ly, rx, ry = left[0] / width, left[1] / height, right[0] / width, right[1] / height
    if not all(0 <= v <= 1 for v in (lx, ly, rx, ry)):
        return None  # 프레임 밖으로 추정된 어깨는 "안 보임"(MediaPipe 와 같은 규칙)
    return Shoulders(cx=(lx + rx) / 2, cy=(ly + ry) / 2, width=math.hypot(lx - rx, ly - ry))


def webcam_from_keypoints(
    persons_kps: Sequence[np.ndarray],
    persons_scores: Sequence[np.ndarray],
    width: int,
    height: int,
    *,
    face_threshold: float,
    yaw_scale: float,
    pitch_scale: float,
) -> WebcamFeatures:
    """사람별 키포인트 -> 웹캠 특징값. 얼굴 수 = 얼굴 점수가 기준값 이상인 사람 수. 각도·어깨는 얼굴이 가장 확실한 사람 기준."""
    face_scores = [face_score(s) for s in persons_scores]
    count = sum(1 for s in face_scores if s >= face_threshold)
    if not persons_kps:
        return WebcamFeatures(face_count=0)
    main = int(np.argmax(face_scores))
    shoulders = shoulders_from_keypoints(persons_kps[main], persons_scores[main], width, height)
    if count == 0:
        return WebcamFeatures(face_count=0, shoulders=shoulders)
    pose = head_pose(persons_kps[main][FACE_START:FACE_END], width, height, yaw_scale, pitch_scale)
    yaw, pitch = pose if pose is not None else (None, None)
    return WebcamFeatures(face_count=count, yaw=yaw, pitch=pitch, shoulders=shoulders)
