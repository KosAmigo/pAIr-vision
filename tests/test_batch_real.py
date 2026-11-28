"""실제 모델 묶음 추론(RealBatchBackend) 통합 시험(CPU): 여러 응시자를 묶어 돌린 결과가 한 장씩 돌린 결과와 같은지,
사람 상자 재사용·주기·fp16(CPU 에서는 꺼짐)을 본다. 가중치·torch 가 있을 때만 돈다(사진은 VISION_TEST_PHOTOS 가 있으면 그 사진,
없으면 합성 이미지)."""

import io
import os
from pathlib import Path

import pytest

from test_real_models import MODELS_DIR, ROOT, _weights_ready

PHOTOS = Path(os.environ["VISION_TEST_PHOTOS"]) if os.environ.get("VISION_TEST_PHOTOS") else None
pytestmark = pytest.mark.skipif(not _weights_ready(), reason="모델 가중치 또는 torch/transformers/rtmlib 없음")


def jpeg(i: int, size=(512, 384), quality=85) -> bytes:
    from PIL import Image, ImageDraw

    if PHOTOS is not None and PHOTOS.is_dir():
        names = sorted(p.name for p in PHOTOS.glob("*.png"))
        img = Image.open(PHOTOS / names[i % len(names)]).convert("RGB")
        img.thumbnail(size)
    else:
        img = Image.new("RGB", size, (200 - i * 5, 200, 190))
        ImageDraw.Draw(img).rectangle((120, 200, 380, 330), fill=(40, 40, 40))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


@pytest.fixture(scope="module")
def rules():
    from pair_vision.rules.config import load_rules

    return load_rules(ROOT / "configs" / "rules.default.yaml")


@pytest.fixture(scope="module")
def single(rules):
    from pair_vision.models.real import RealDetector

    det = RealDetector(MODELS_DIR, rules.models, verify=False)
    det.load()
    return det


@pytest.fixture(scope="module")
def batched(rules):
    from pair_vision.models.batch import BatchedDetector
    from pair_vision.models.batch_real import RealBatchBackend

    sizes = []
    det = BatchedDetector(RealBatchBackend(MODELS_DIR, rules.models, verify=False), max_batch=8, wait_ms=50)
    det.batch_observer = lambda stage, n: sizes.append((stage, n))
    det.load()
    det.sizes = sizes
    yield det
    det.close()


def same_boxes(a, b, tol=0.02):
    if a is None or b is None:
        return a is b
    return all(abs(x - y) <= tol for x, y in zip((a.x0, a.y0, a.x1, a.y1), (b.x0, b.y0, b.x1, b.y1)))


def test_cpu_keeps_fp32(batched, single):
    import torch

    assert single.precision == batched._backend.precision == {"dfine": "fp32", "owlv2": "fp32"}
    assert next(single._owl.parameters()).dtype == torch.float32 and next(single._dfine.parameters()).dtype == torch.float32


def test_batched_phone_results_match_one_by_one(batched, single):
    from pair_vision.models.base import Frame

    frames = [Frame(f"P{i}", "phone", 1, 1000, jpeg(i)) for i in range(6)]
    got = batched.run(frames)
    want = [single.run([f])[0] for f in frames]
    assert [d.key for d in got] == [f.key for f in frames]
    for g, w in zip(got, want):
        assert same_boxes(g.keyboard, w.keyboard) and same_boxes(g.mouse, w.mouse)
        assert g.hand_count == w.hand_count
        assert {o.label for o in g.objects} == {o.label for o in w.objects}
        for gh, wh in zip(g.hands, w.hands):
            assert abs(gh.x - wh.x) < 0.01 and abs(gh.y - wh.y) < 0.01
    stages = {s for s, _ in batched.sizes}
    assert {"dfine", "rtmw", "owlv2"} <= stages
    assert max(n for s, n in batched.sizes if s == "rtmw") > 1  # 실제로 여러 응시자가 한 묶음으로 돌았다


def test_cadences_and_person_box_reuse(batched):
    from pair_vision.models.base import Frame

    before = dict(batched.stats())
    calls = lambda stage: batched.stats().get(stage, (0, 0))[0] - before.get(stage, (0, 0))[0]  # noqa: E731
    web = jpeg(1, size=(400, 300), quality=55)
    for t in (0, 1000, 2000):  # 3초 안: 사람 상자는 처음 한 번만 찾는다
        d = batched.run([Frame("W", "webcam", t + 1, t, web)])[0]
        assert d.webcam is not None
    assert calls("person") == 1 and calls("rtmw") == 3
    batched.run([Frame("W", "webcam", 9, 3000, web)])  # 3초 지남 -> 다시 찾는다
    assert calls("person") == 2
    for t in (0, 1000, 2000, 3000):  # D-FINE 3초 주기, OWLv2 10초 주기
        batched.run([Frame("C", "phone", t + 1, t, jpeg(2))])
    assert calls("dfine") == 2 and calls("owlv2") == 1
    batched.forget("C")
    batched.run([Frame("C", "phone", 9, 3500, jpeg(2))])
    assert calls("owlv2") == 2  # 세션이 끝나면 주기 상태도 지운다


def test_webcam_features_match_one_by_one(batched, single):
    from pair_vision.models.base import Frame

    web = jpeg(3, size=(400, 300), quality=55)
    g = batched.run([Frame("W2", "webcam", 1, 1000, web)])[0].webcam
    w = single.run([Frame("W2", "webcam", 1, 1000, web)])[0].webcam
    assert g.face_count == w.face_count
    if g.yaw is not None:
        assert abs(g.yaw - w.yaw) < 1.0 and abs(g.pitch - w.pitch) < 1.0


def test_broken_jpeg_fails_only_that_frame(batched):
    from pair_vision.models.base import Frame

    good = Frame("G", "phone", 1, 1000, jpeg(0))
    bad = Frame("B", "phone", 1, 1000, b"\xff\xd8\xff not a jpeg")
    out = batched.run([bad, good])
    assert [d.key for d in out] == ["G"]


@pytest.fixture(scope="module")
def on_device(rules):
    """전부 GPU 경로(JPEG 풀기·전처리·RTMW 자르기·YOLOX 레터박스를 torch 로)를 CPU 에서 강제로 탄다."""
    from pair_vision.models.batch import BatchedDetector
    from pair_vision.models.batch_real import RealBatchBackend

    det = BatchedDetector(RealBatchBackend(MODELS_DIR, rules.models, verify=False, device_preprocess=True),
                          max_batch=8, wait_ms=50)
    det.load()
    yield det
    det.close()


def test_device_preprocess_path_matches_cpu_path(on_device, batched):
    from pair_vision.models.base import Frame

    assert on_device._backend.device_preprocess and not batched._backend.device_preprocess
    phones = [Frame(f"D{i}", "phone", 1, 1000, jpeg(i)) for i in range(4)]
    same = [Frame("E" + f.key, f.channel, f.seq, f.recv_ms, f.jpeg) for f in phones]  # 같은 사진, 다른 세션
    for g, w in zip(on_device.run(phones), batched.run(same), strict=True):
        assert same_boxes(g.keyboard, w.keyboard, 0.03) and same_boxes(g.mouse, w.mouse, 0.03)
        assert g.hand_count == w.hand_count
        assert {o.label for o in g.objects} == {o.label for o in w.objects}
        for gh, wh in zip(g.hands, w.hands):
            assert abs(gh.x - wh.x) < 0.02 and abs(gh.y - wh.y) < 0.02
    web = jpeg(3, size=(400, 300), quality=55)
    g = on_device.run([Frame("DW", "webcam", 1, 1000, web)])[0].webcam
    w = batched.run([Frame("EW", "webcam", 1, 1000, web)])[0].webcam
    assert g.face_count == w.face_count
    if g.yaw is not None and w.yaw is not None:
        assert abs(g.yaw - w.yaw) < 3.0 and abs(g.pitch - w.pitch) < 3.0
    stats = on_device.stats()
    assert stats["decode"][0] >= 5  # 풀기가 엔진 단계로 돌았다
    out = on_device.run([Frame("BAD", "phone", 1, 1000, b"\xff\xd8\xff broken"), Frame("OK", "phone", 1, 1000, jpeg(0))])
    assert [d.key for d in out] == ["OK"]  # 깨진 JPEG 은 그 프레임만 버린다
