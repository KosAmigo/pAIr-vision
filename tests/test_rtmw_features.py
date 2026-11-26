"""RTMW(COCO-WholeBody) 결과 -> 손·웹캠 특징값 시험. 가상의 키포인트로 계산을 확인한다(모델·GPU 없이)."""

import math

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from pair_vision.models.rtmw_features import (  # noqa: E402
    FACE_MODEL_3D,
    FACE_PNP_IDX,
    FACE_START,
    hands_from_keypoints,
    head_pose,
    head_pose_raw,
    shoulders_from_keypoints,
    webcam_from_keypoints,
)
from pair_vision.rules.config import parse_rules  # noqa: E402

W, H = 400, 300


def blank():
    return np.zeros((133, 2)), np.zeros(133)


def put_hand(kps, scores, start, x, y, score):
    kps[start : start + 21] = (x, y)
    kps[start + 9] = (x + 20, y + 10)  # 가운뎃손가락 시작 관절
    scores[start : start + 21] = score


def synth_face(kps, yaw_deg, pitch_deg, tz=800.0):
    """일반 얼굴 6점 모델을 yaw·pitch 만큼 돌려 화면에 투영해 얼굴 키포인트로 넣는다(PnP 가 되돌려야 하는 값)."""
    ry, rx = math.radians(yaw_deg), math.radians(pitch_deg)
    rot_y = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    rot_x = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    rvec, _ = cv2.Rodrigues(rot_y @ rot_x)
    cam = np.array([[W, 0, W / 2], [0, W, H / 2], [0, 0, 1]], dtype=np.float64)
    pts, _ = cv2.projectPoints(FACE_MODEL_3D, rvec, np.array([0.0, 0.0, tz]), cam, None)
    for idx, p in zip(FACE_PNP_IDX, pts.reshape(-1, 2), strict=True):
        kps[FACE_START + idx] = p


def test_hands_use_score_and_wrist_mcp_midpoint():
    kps, scores = blank()
    put_hand(kps, scores, 91, 100, 150, 0.9)    # 왼손
    put_hand(kps, scores, 112, 300, 200, 0.35)  # 오른손: 점수가 낮아 안 보임
    hands = hands_from_keypoints(kps, scores, W, H, threshold=0.5)
    assert [(h.side, round(h.x, 3), round(h.y, 3)) for h in hands] == [("left", round((100 + 120) / 2 / W, 3), round((150 + 160) / 2 / H, 3))]
    assert hands[0].score == pytest.approx(0.9)


def test_two_hands_sorted_by_score():
    kps, scores = blank()
    put_hand(kps, scores, 91, 50, 50, 0.6)
    put_hand(kps, scores, 112, 250, 50, 0.85)
    assert [h.side for h in hands_from_keypoints(kps, scores, W, H, 0.5)] == ["right", "left"]


@pytest.mark.parametrize(("yaw", "pitch"), [(0, 0), (15, 0), (-15, 0), (0, 15), (0, -10)])
def test_head_pose_recovers_synthetic_rotation(yaw, pitch):
    kps, _ = blank()
    synth_face(kps, yaw, pitch)
    got_yaw, got_pitch, _ = head_pose_raw(kps[FACE_START:], W, H)
    assert abs(abs(got_yaw) - abs(yaw)) < 2.0 and abs(got_pitch - pitch) < 2.0


def test_head_pose_signs_match_mediapipe_convention_and_scale():
    kps, _ = blank()
    synth_face(kps, 20, 12)
    raw_yaw = head_pose_raw(kps[FACE_START:], W, H)[0]
    yaw, pitch = head_pose(kps[FACE_START:], W, H, yaw_scale=0.5, pitch_scale=1.0)
    assert yaw == pytest.approx(-raw_yaw * 0.5)  # 부호는 MediaPipe 와 맞추고 크기는 보정 계수만큼
    assert pitch == pytest.approx(12, abs=2)  # 숙이면 +


def test_shoulders_need_score_and_inside_frame():
    kps, scores = blank()
    kps[5], kps[6] = (100, 250), (300, 250)
    scores[5] = scores[6] = 0.9
    sh = shoulders_from_keypoints(kps, scores, W, H)
    assert sh is not None and sh.cx == pytest.approx(0.5) and sh.width == pytest.approx(0.5)
    scores[6] = 0.3
    assert shoulders_from_keypoints(kps, scores, W, H) is None
    scores[6] = 0.9
    kps[6] = (300, 400)  # 프레임 밖
    assert shoulders_from_keypoints(kps, scores, W, H) is None


def person(face_score, with_pose=True):
    kps, scores = blank()
    scores[23:91] = face_score
    scores[5] = scores[6] = 0.9
    kps[5], kps[6] = (150, 250), (250, 250)
    if with_pose:
        synth_face(kps, 5, 3)
    return kps, scores


def test_webcam_counts_faces_and_uses_main_person():
    (k1, s1), (k2, s2) = person(0.98), person(0.7)
    f = webcam_from_keypoints([k1, k2], [s1, s2], W, H, face_threshold=0.5, yaw_scale=0.4, pitch_scale=1.0)
    assert f.face_count == 2 and f.yaw is not None and f.shoulders is not None


def test_webcam_without_face_or_person():
    k, s = person(0.12)
    f = webcam_from_keypoints([k], [s], W, H, face_threshold=0.5, yaw_scale=0.4, pitch_scale=1.0)
    assert f.face_count == 0 and f.yaw is None and f.shoulders is not None
    empty = webcam_from_keypoints([], [], W, H, face_threshold=0.5, yaw_scale=0.4, pitch_scale=1.0)
    assert empty.face_count == 0 and empty.shoulders is None


def test_rtmw_config_defaults_and_validation():
    base = {"version": "v", "profiles": {"default": {}}}
    m = parse_rules(base).models
    assert (m.vision_backend, m.face_threshold, m.yaw_scale, m.pitch_scale) == ("rtmw", 0.5, 0.55, 1.0)
    assert parse_rules({**base, "models": {"vision_backend": "mediapipe"}}).models.vision_backend == "mediapipe"
    with pytest.raises(ValueError, match="vision_backend"):
        parse_rules({**base, "models": {"vision_backend": "yolo"}})
    with pytest.raises(ValueError, match="yaw_scale"):
        parse_rules({**base, "models": {"yaw_scale": 0}})
