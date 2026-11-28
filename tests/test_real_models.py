"""실제 모델 통합 시험. 가중치(VISION_MODELS_DIR, 기본 ./models)와 torch·transformers·mediapipe 가 있고,
사진 폴더(VISION_TEST_PHOTOS, 1.png~)가 있을 때만 돈다. 없으면 건너뛴다(CPU 도커·CI 는 가짜 모델 시험만).

사진은 저장소에 없다(개인 정보 가능성). 8장은 같은 장면의 변형이라 이 시험은 "배선이 맞다"는 확인이지 정확도 측정이 아니다.
"""

import io
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("VISION_MODELS_DIR", ROOT / "models"))
PHOTOS = Path(os.environ["VISION_TEST_PHOTOS"]) if os.environ.get("VISION_TEST_PHOTOS") else None


def _weights_ready() -> bool:
    needed = ("dfine-small-coco", "owlv2-base-patch16-ensemble", "rtmw/yolox_tiny.onnx", "rtmw/rtmw-dw-l-m_256x192.onnx")
    if not all((MODELS_DIR / n).exists() for n in needed):
        return False
    try:
        import onnxruntime  # noqa: F401
        import rtmlib  # noqa: F401
        import PIL  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = [
    pytest.mark.skipif(not _weights_ready(), reason="모델 가중치 또는 torch/transformers/mediapipe 없음"),
    pytest.mark.skipif(PHOTOS is None or not PHOTOS.is_dir(), reason="VISION_TEST_PHOTOS 사진 폴더 없음"),
]


def jpeg_of(name: str) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.open(PHOTOS / name).convert("RGB").save(buf, "JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture(scope="module")
def detector():
    from pair_vision.models.real import RealDetector
    from pair_vision.rules.config import load_rules

    det = RealDetector(MODELS_DIR, load_rules(ROOT / "configs" / "rules.default.yaml").models, lock_path=ROOT / "configs" / "models.lock")
    det.load()  # sha256 검증 포함
    return det


def test_phone_photo_gives_keyboard_mouse_and_at_most_two_rule_hands(detector):
    from pair_vision.models.base import Frame

    for name in sorted(p.name for p in PHOTOS.glob("*.png"))[:3]:
        det = detector.run([Frame("K", "phone", 1, 1000, jpeg_of(name))])[0]
        assert det.keyboard is not None and det.mouse is not None
        assert len(det.hands) <= 2 and (det.hand_count or 0) >= len(det.hands)
        assert all(0 <= h.x <= 1 and 0 <= h.y <= 1 for h in det.hands)


def test_owl_runs_only_on_interval_and_forget_resets(detector):
    from pair_vision.models.base import Frame

    jpeg = jpeg_of("1.png")
    labels = lambda t: {o.label for o in detector.run([Frame("OWL", "phone", t, t, jpeg)])[0].objects}  # noqa: E731
    first = labels(0)
    assert "tablet" in first or "smartwatch" in first or "phone" in first  # 첫 프레임은 OWLv2 실행
    assert not (labels(1000) & {"tablet", "smartwatch", "phone"})  # 10초 안: 실행 안 함
    assert labels(10000) & {"tablet", "smartwatch", "phone"}
    detector.forget("OWL")
    assert labels(10100) & {"tablet", "smartwatch", "phone"}


def test_webcam_channel_gives_face_features(detector):
    from pair_vision.models.base import Frame

    # 책상 사진에는 얼굴이 없다: 얼굴 0개(NO_FACE 조건)이고 손·물체 결과는 비어 있다.
    det = detector.run([Frame("K", "webcam", 1, 1000, jpeg_of("3.png"))])[0]
    assert det.webcam is not None and det.webcam.face_count == 0 and det.webcam.yaw is None
    assert det.keyboard is None and det.hands == () and det.objects == ()


def test_phone_hand_counts_match_the_eight_photos(detector):
    """같은 장면 8장에서 손을 눈으로 센 값(5번 사진은 손 3개지만 RTMW 는 한 상자에서 둘까지). 정확도 확정이 아니라 배선 확인."""
    from pair_vision.models.base import Frame

    counts = []
    for name in sorted(p.name for p in PHOTOS.glob("*.png"))[:8]:
        det = detector.run([Frame("K", "phone", 1, 1000, jpeg_of(name))])[0]
        counts.append(det.hand_count)
    assert counts == [2, 2, 2, 2, 2, 2, 1, 1]


def test_broken_jpeg_is_skipped_not_reported_as_empty(detector):
    from pair_vision.models.base import Frame

    good = Frame("K", "phone", 1, 1000, jpeg_of("1.png"))
    bad = Frame("K2", "phone", 1, 1000, b"\xff\xd8\xff not a jpeg")
    out = detector.run([bad, good])
    assert [d.key for d in out] == ["K"]  # 실패 프레임은 "손 없음"이 아니라 결과 없음
    with pytest.raises(RuntimeError):
        detector.run([bad])


def test_session_makes_tablet_and_smartwatch_events_from_real_photo(detector, rules):
    from pair_vision.models.base import Frame
    from pair_vision.session.session import Session

    jpeg = jpeg_of("1.png")  # 1번 사진: 태블릿·스마트워치가 보인다
    session = Session("K", "exam", 0, rules.profile("default"))
    types = []
    for t in range(0, 32000, 1000):  # 스마트워치 30초 연속(OWLv2 10초 주기 4번)
        det = detector.run([Frame("S", "phone", t, t, jpeg)])[0]
        types += [e.type for e in session.on_detection(det).events if not e.final]
    assert "TABLET" in types and "SMARTWATCH" in types


WEBCAM_PHOTOS = (
    Path(os.environ["VISION_TEST_WEBCAM_PHOTOS"]) if os.environ.get("VISION_TEST_WEBCAM_PHOTOS") else None
)


@pytest.mark.skipif(WEBCAM_PHOTOS is None or not WEBCAM_PHOTOS.is_dir(), reason="VISION_TEST_WEBCAM_PHOTOS 폴더 없음")
def test_frontal_webcam_photos_give_one_face_near_zero_pose(detector):
    """얼굴이 나온 정면 웹캠 사진(본인 동의, git 에 없음): 얼굴 1개, 각도는 0 근처, 어깨가 잡힌다."""
    from PIL import Image

    from pair_vision.models.base import Frame

    seen = 0
    for path in sorted(WEBCAM_PHOTOS.glob("*.jpg")):
        img = Image.open(path).convert("RGB")
        img.thumbnail((400, 400))  # 계약의 웹캠 maxEdge
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=55)
        f = detector.run([Frame("W", "webcam", 1, 1000, buf.getvalue())])[0].webcam
        # RTMW 는 고개를 숙이거나 많이 돌린 사진에서도 얼굴을 잡는다(MediaPipe 는 놓쳐 NO_FACE 로 셌다)
        assert f.face_count == 1 and f.yaw is not None and f.shoulders is not None
        seen += 1
    assert seen >= 1
