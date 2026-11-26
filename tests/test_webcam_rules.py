"""웹캠 규칙(프론트 pcRules.ts 포팅) 시험: 조건 판정, 보정, 이벤트, 특징값 변환. 모델 없이 돈다."""

import math
from types import SimpleNamespace

import pytest

from pair_vision.evidence.events import SOURCE_BY_TYPE
from pair_vision.models.base import Detection, Shoulders, WebcamFeatures
from pair_vision.models.fake import WEBCAM_STATES
from pair_vision.models.webcam_features import (
    gaze_from_blendshapes,
    head_pose_from_matrix,
    shoulders_from_pose,
    webcam_features,
)
from pair_vision.rules.config import parse_rules
from pair_vision.rules.webcam import (
    WebcamBaseline,
    WebcamThresholds,
    angle_diff,
    build_baseline,
    webcam_conditions,
)
from pair_vision.session.session import Session

T = WebcamThresholds()
SH = Shoulders(0.5, 0.7, 0.4)
BASE = WebcamBaseline(yaw=2.0, pitch=5.0, gaze_x=0.0, gaze_y=0.0, shoulders=SH)


def feat(**over):
    d = dict(face_count=1, yaw=2.0, pitch=5.0, gaze_x=0.0, gaze_y=0.0, blink=0.0, shoulders=SH)
    d.update(over)
    return WebcamFeatures(**d)


def test_front_pose_has_no_active_condition():
    assert not any(webcam_conditions(feat(), BASE, T).values())


@pytest.mark.parametrize(
    ("over", "label"),
    [
        ({"face_count": 0}, "no_face"),
        ({"face_count": 2}, "multi_face"),
        ({"yaw": 40.0}, "head_turn"),
        ({"pitch": -20.0}, "head_turn"),  # 고개를 20도 넘게 듦
        ({"pitch": 40.0}, "head_down"),
        ({"gaze_x": 0.6}, "gaze_away"),
        ({"gaze_y": 0.6}, "gaze_away"),
        ({"shoulders": Shoulders(0.7, 0.7, 0.4)}, "torso_shift"),
        ({"shoulders": Shoulders(0.5, 0.7, 0.2)}, "torso_turn"),
        ({"shoulders": Shoulders(0.5, 0.7, 0.6)}, "torso_turn"),
        ({"shoulders": None}, "torso_shift"),
    ],
)
def test_each_condition_triggers(over, label):
    assert webcam_conditions(feat(**over), BASE, T)[label] is True


def test_blink_suppresses_gaze_and_no_face_has_no_head_conditions():
    assert "gaze_away" not in webcam_conditions(feat(gaze_x=0.9, blink=0.8), BASE, T)
    cond = webcam_conditions(feat(face_count=0, yaw=None, pitch=None), BASE, T)
    assert "head_turn" not in cond and cond["no_face"]


def test_angle_diff_wraps():
    assert angle_diff(179, -179) == -2
    assert angle_diff(-179, 179) == 2


def test_baseline_median_and_rejects_poor_samples():
    samples = [feat(yaw=y) for y in (1.0, 3.0, 2.0, 100.0)]
    b = build_baseline(samples, min_samples=3)
    assert b is not None and b.yaw == 2.5 and b.shoulders == SH
    assert build_baseline([feat(), feat(face_count=0, yaw=None, pitch=None)], min_samples=3) is None
    assert build_baseline([feat(), feat(), feat(face_count=2)], min_samples=3) is None  # 정면 1얼굴 3개 미만
    no_sh = build_baseline([feat(shoulders=None)] * 4, min_samples=3)
    assert no_sh is not None and no_sh.shoulders is None


def test_head_pose_from_identity_and_rotation():
    ident = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    assert head_pose_from_matrix(ident) == (0.0, 0.0, 0.0)
    a = math.radians(30)  # x축 회전 30도: pitch 30 (프론트와 같은 식: atan2(r21, r22))
    pitch_m = [[1, 0, 0, 0], [0, math.cos(a), -math.sin(a), 0], [0, math.sin(a), math.cos(a), 0], [0, 0, 0, 1]]
    yaw, pitch, _ = head_pose_from_matrix(pitch_m)
    assert round(pitch) == 30 and round(yaw) == 0
    yaw_m = [[math.cos(a), 0, math.sin(a), 0], [0, 1, 0, 0], [-math.sin(a), 0, math.cos(a), 0], [0, 0, 0, 1]]
    assert round(head_pose_from_matrix(yaw_m)[0]) == 30  # asin(-r20) = asin(sin a)


def test_gaze_from_blendshapes_directions():
    right = {"eyeLookInLeft": 0.8, "eyeLookOutRight": 0.8}
    gx, gy, blink = gaze_from_blendshapes(right)
    assert gx == pytest.approx(0.8) and gy == 0 and blink == 0
    down = gaze_from_blendshapes({"eyeLookDownLeft": 0.6, "eyeLookDownRight": 0.4, "eyeBlinkLeft": 1.0})
    assert down[1] == pytest.approx(0.5) and down[2] == pytest.approx(0.5)


def test_shoulders_need_both_visible_inside_frame():
    lms = [SimpleNamespace(x=0.0, y=0.0, visibility=1.0) for _ in range(13)]
    lms[11] = SimpleNamespace(x=0.4, y=0.7, visibility=0.9)
    lms[12] = SimpleNamespace(x=0.6, y=0.7, visibility=0.9)
    sh = shoulders_from_pose(lms)
    assert sh is not None and sh.cx == pytest.approx(0.5) and sh.width == pytest.approx(0.2)
    lms[12] = SimpleNamespace(x=0.6, y=0.7, visibility=0.2)
    assert shoulders_from_pose(lms) is None
    assert shoulders_from_pose(None) is None and shoulders_from_pose(lms[:5]) is None


def test_webcam_features_without_face_skips_head_and_gaze():
    f = webcam_features(0, [[1, 0, 0, 0]] * 4, {"eyeBlinkLeft": 1.0}, None)
    assert f.face_count == 0 and f.yaw is None and f.gaze_x is None and f.shoulders is None


# --- 세션: 보정 -> 이벤트 ---
def det(t, name="FRONT"):
    return Detection("K", "webcam", 1, t, webcam=WEBCAM_STATES[name])


def exam(rules):
    return Session("K", "exam", 0, rules.profile("default"))


def calibrate(s, start=0):
    for t in range(start, start + 4000, 1000):
        s.on_detection(det(t))
    assert s.calibration == "done"
    return start + 4000


def test_calibration_completes_with_front_faces_and_events_follow(rules):
    s = exam(rules)
    assert s.on_detection(det(0)).events == [] and s.calibration == "pending"
    t = calibrate(s)
    events = []
    for k in range(4):  # 얼굴 없음 3초 -> NO_FACE(2초 이상)
        events += s.on_detection(det(t + k * 1000, "NO_FACE")).events
    assert [(e.type, e.final) for e in events] == [("NO_FACE", False)]
    assert SOURCE_BY_TYPE["NO_FACE"] == "PC_CAM"
    closing = s.on_detection(det(t + 10000)).events  # 정상으로 돌아와 끊김 허용을 넘김 -> final
    assert [(e.type, e.final) for e in closing] == [("NO_FACE", True)]
    assert s.end() == []


@pytest.mark.parametrize(
    ("state", "etype", "seconds"),
    [("MULTI_FACE", "MULTI_FACE", 2), ("HEAD_TURN", "HEAD_TURN", 3), ("HEAD_DOWN", "HEAD_DOWN", 6),
     ("TORSO_SHIFT", "TORSO_SHIFT", 3), ("TORSO_TURN", "TORSO_TURN", 3)],
)
def test_each_rule_fires_after_its_duration(rules, state, etype, seconds):
    s = exam(rules)
    t = calibrate(s)
    events = []
    for k in range(seconds + 1):
        events += s.on_detection(det(t + k * 1000, state)).events
    assert etype in [e.type for e in events]


def test_gaze_away_is_disabled_by_default_but_can_be_enabled(rules):
    assert "gaze_away" not in rules.profile("default").webcam.rules
    s = exam(rules)
    t = calibrate(s)
    assert all(s.on_detection(det(t + k * 1000, "GAZE_AWAY")).events == [] for k in range(8))
    on = parse_rules({"version": "v", "profiles": {"default": {"webcam": {"rules": {"gaze_away": {"enabled": True}}}}}})
    s2 = Session("K", "exam", 0, on.profile("default"))
    t2 = calibrate(s2)
    events = []
    for k in range(7):
        events += s2.on_detection(det(t2 + k * 1000, "GAZE_AWAY")).events
    assert [e.type for e in events] == ["GAZE_AWAY"]


def test_no_event_before_duration_and_none_in_check_mode(rules):
    s = exam(rules)
    t = calibrate(s)
    assert s.on_detection(det(t, "HEAD_DOWN")).events == []
    assert s.on_detection(det(t + 1000, "HEAD_DOWN")).events == []  # 5초 전
    check = Session("K", "check", 0, rules.profile("default"))
    assert all(check.on_detection(det(i * 1000, "NO_FACE")).events == [] for i in range(10))


def test_calibration_retries_when_no_front_face(rules):
    s = exam(rules)
    for t in range(0, 5000, 1000):
        s.on_detection(det(t, "NO_FACE"))
    assert s.calibration == "pending"  # 정면 얼굴이 안 잡히면 끝내지 않고 다시 모은다
    calibrate(s, start=5000)


def test_frames_without_features_never_count_as_anomaly(rules):
    s = exam(rules)
    t = calibrate(s)
    blank = Detection("K", "webcam", 1, t + 1000)  # 웹캠 모델이 없는 프레임
    assert all(s.on_detection(Detection("K", "webcam", 1, t + i * 1000)).events == [] for i in range(10))
    assert blank.webcam is None


def test_recalibrate_resets_webcam_state(rules):
    s = exam(rules)
    t = calibrate(s)
    for k in range(3):
        s.on_detection(det(t + k * 1000, "NO_FACE"))
    s.recalibrate()
    assert s.calibration == "pending" and s.end() == []


def test_webcam_rule_config_validation():
    base = {"version": "v", "profiles": {"default": {}}}
    with pytest.raises(ValueError, match="알 수 없는"):
        parse_rules({"version": "v", "profiles": {"default": {"webcam": {"rules": {"cat": {}}}}}})
    with pytest.raises(ValueError, match="yaw_deg"):
        parse_rules({"version": "v", "profiles": {"default": {"webcam": {"thresholds": {"yaw_deg": 0}}}}})
    assert parse_rules(base).profile("default").webcam.rules["head_down"].min_ms == 5000
