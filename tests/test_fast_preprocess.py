"""torch 전처리(models.fast_preprocess)가 transformers 기본 전처리와 같은 값을 내는지(가장자리·반올림 차이만) 확인한다.
torch·transformers 와 모델 폴더의 전처리 설정 파일이 있을 때만 돈다."""

import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("VISION_MODELS_DIR", ROOT / "models"))
OWL = MODELS / "owlv2-base-patch16-ensemble"
DFINE = MODELS / "dfine-small-coco"
pytestmark = pytest.mark.skipif(not (OWL / "preprocessor_config.json").is_file(), reason="모델 폴더 없음")


def photo(h: int, w: int, seed: int = 0) -> np.ndarray:
    """실제 사진처럼 부드러운 영역 + 경계 + 잡음이 섞인 합성 이미지."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([(xx * 255 / w), (yy * 255 / h), ((xx + yy) % 97) * 2.5], axis=-1)
    img[h // 3 : h // 2, w // 4 : w // 2] = (30, 40, 50)
    img += rng.normal(0, 12, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("shape", [(384, 512), (300, 400), (512, 384), (1080, 1440)])
def test_owl_pixels_match_reference_processor(shape):
    from transformers import AutoImageProcessor

    from pair_vision.models.preprocess import owl_pixels

    rgb = photo(*shape)
    ref = AutoImageProcessor.from_pretrained(OWL)(images=rgb, return_tensors="pt")["pixel_values"][0]
    got = owl_pixels(rgb)
    assert got.shape == ref.shape == (3, 960, 960)
    inner = (slice(None), slice(4, -4), slice(4, -4))  # 가장자리 몇 픽셀은 보간 경계 처리 방식이 다르다
    diff = (got - ref)[inner].abs()
    assert float(diff.mean()) < 0.005 and float(diff.quantile(0.999)) < 0.1


@pytest.mark.parametrize("shape", [(384, 512), (300, 400)])
def test_dfine_pixels_match_reference_processor(shape):
    from transformers import AutoImageProcessor

    from pair_vision.models.preprocess import dfine_pixels

    rgb = photo(*shape, seed=1)
    ref = AutoImageProcessor.from_pretrained(DFINE)(images=rgb, return_tensors="pt")["pixel_values"][0]
    got = dfine_pixels(rgb)
    assert got.shape == ref.shape == (3, 640, 640)
    diff = (got - ref).abs()
    assert float(diff.mean()) < 0.005 and float(diff.max()) < 0.05


def _jpeg(rgb, quality=80) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def test_decode_jpegs_matches_pil_and_fails_on_broken_data():
    import io

    from PIL import Image

    from pair_vision.models.preprocess import decode_jpegs

    data = _jpeg(photo(384, 512, seed=3))
    ref = torch.from_numpy(np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))).permute(2, 0, 1)
    got = decode_jpegs([data, data])
    assert got[0].shape == (3, 384, 512) and got[0].dtype == torch.uint8
    assert float((got[0].float() - ref.float()).abs().mean()) < 1.0  # 디코더마다 IDCT 반올림만 다르다
    with pytest.raises(Exception):
        decode_jpegs([b"\xff\xd8\xff not a jpeg"])


@pytest.mark.parametrize("box", [(0, 0, 512, 384), (100, 50, 300, 360), (-20, 10, 200, 500)])
def test_warp_crops_match_rtmlib_preprocess(box):
    pytest.importorskip("rtmlib")
    cv2 = pytest.importorskip("cv2")  # noqa: F841 - rtmlib 이 쓴다
    from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs, top_down_affine

    from pair_vision.models.preprocess import RTMW_MEAN, RTMW_STD, crop_params, warp_crops

    rgb = photo(384, 512, seed=4)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    center, scale = bbox_xyxy2cs(np.array(box), padding=1.25)
    ref_img, ref_scale = top_down_affine((192, 256), scale, center, bgr)
    ref = (ref_img - np.array(RTMW_MEAN)) / np.array(RTMW_STD)  # rtmlib RTMPose.preprocess 와 같은 계산
    c, s, mat = crop_params(box)
    assert np.allclose(c, center) and np.allclose(s, ref_scale)
    got = warp_crops(torch.from_numpy(rgb).permute(2, 0, 1), [(c, s, mat)])[0].numpy().transpose(1, 2, 0)
    diff = np.abs(got - ref)[2:-2, 2:-2]
    assert got.shape == ref.shape and diff.mean() < 0.01 and np.quantile(diff, 0.999) < 0.1


@pytest.mark.parametrize("shape", [(300, 400), (384, 512), (720, 1280)])
def test_letterbox_matches_yolox_preprocess(shape):
    pytest.importorskip("rtmlib")
    import cv2

    from pair_vision.models.preprocess import letterbox

    rgb = photo(*shape, seed=5)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    ratio = min(416 / shape[0], 416 / shape[1])
    ref = np.ones((416, 416, 3), dtype=np.uint8) * 114
    resized = cv2.resize(bgr, (int(shape[1] * ratio), int(shape[0] * ratio)), interpolation=cv2.INTER_LINEAR)
    ref[: resized.shape[0], : resized.shape[1]] = resized
    got, r = letterbox(torch.from_numpy(rgb).permute(2, 0, 1))
    assert abs(r - ratio) < 1e-9
    diff = np.abs(got[0].numpy().transpose(1, 2, 0) - ref.astype(np.float32))
    tol = 2.0 if ratio >= 1 else 6.0  # 축소는 cv2(INTER_LINEAR, 안티에일리어싱 없음)와 반올림 방식이 조금 다르다
    assert diff.mean() < tol
