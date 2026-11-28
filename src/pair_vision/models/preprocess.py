"""D-FINE·OWLv2 이미지 전처리를 torch 로(가능하면 GPU 에서) 한다 (`models.fast_preprocess`).

transformers 기본(PIL·scipy) 전처리는 OWLv2 가 CPU 에서 장당 약 120ms(960px 정사각형으로 패딩·리사이즈, scipy zoom)라
GPU 추론(약 155ms)과 맞먹었다(용량 점검 C). 같은 계산을 torch 로 옮긴다:
- OWLv2: 0~1 로 나눔 -> 오른쪽·아래를 0 으로 채워 정사각형 -> 960x960 양선형(축소면 안티에일리어싱) -> CLIP 평균·표준편차 정규화
  (transformers Owlv2ImageProcessorPil: rescale -> pad(0) -> scipy zoom order=1, grid_mode -> normalize 와 같은 순서)
- D-FINE(RTDetr 처리기): 640x640 양선형(PIL BILINEAR 과 같은 안티에일리어싱) -> 0~1, 정규화 없음
가장자리 1픽셀 보간 방식(scipy mirror vs torch clamp)과 반올림 차이만 있다(시험에서 기준 전처리와 수치 비교).
torch 는 선택 의존성이라 함수 안에서만 쓴다.
"""

from __future__ import annotations

import numpy as np

OWL_SIZE = 960
OWL_MEAN = (0.48145466, 0.4578275, 0.40821073)
OWL_STD = (0.26862954, 0.26130258, 0.27577711)
DFINE_SIZE = (640, 640)


def _to_chw(rgb: np.ndarray, device):
    import torch

    arr = np.ascontiguousarray(rgb)
    if not arr.flags.writeable:  # PIL 에서 온 읽기 전용 배열(torch 는 쓰기 가능 배열만 받는다)
        arr = arr.copy()
    t = torch.from_numpy(arr).to(device, non_blocking=True)
    return t.permute(2, 0, 1).float().div_(255.0)


def owl_pixels(rgb, device="cpu", size: int = OWL_SIZE):
    """HxWx3 uint8 RGB(numpy) 또는 (3,H,W) uint8 텐서 -> (3, size, size) float32 텐서(정규화까지)."""
    import torch
    import torch.nn.functional as F

    t = _chw_float(rgb, device)
    _, h, w = t.shape
    side = max(h, w)
    t = F.pad(t, (0, side - w, 0, side - h), value=0.0)
    if side > size:  # 축소: 기준과 같이 가우시안(sigma=(배율-1)/2, scipy truncate 4, mirror 경계)으로 먼저 흐린다
        t = _gaussian_blur(t, (side / size - 1) / 2)
    t = F.interpolate(t[None], size=(size, size), mode="bilinear", align_corners=False)[0]
    mean = torch.tensor(OWL_MEAN, device=t.device).view(3, 1, 1)
    std = torch.tensor(OWL_STD, device=t.device).view(3, 1, 1)
    return (t - mean) / std


def _gaussian_blur(t, sigma: float):
    import torch
    import torch.nn.functional as F

    if sigma <= 0:
        return t
    radius = int(4.0 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, device=t.device, dtype=t.dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    c = t.shape[0]
    t = t[None]
    t = F.conv2d(F.pad(t, (radius, radius, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    t = F.conv2d(F.pad(t, (0, 0, radius, radius), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return t[0]


def dfine_pixels(rgb, device="cpu", size: tuple[int, int] = DFINE_SIZE):
    """HxWx3 uint8 RGB(numpy) 또는 (3,H,W) uint8 텐서 -> (3, 640, 640) float32 텐서(0~1)."""
    import torch.nn.functional as F

    t = _chw_float(rgb, device)
    return F.interpolate(t[None], size=size, mode="bilinear", align_corners=False, antialias=True)[0].clamp_(0.0, 1.0)


# --- 전부 GPU 로(2026-11-27 사용자 요청): JPEG 풀기·RTMW 자르기·YOLOX 레터박스 ---
# 이미지는 (3, H, W) uint8 RGB 텐서(GPU)로 다닌다. 모델 입력·출력(ONNX Runtime)은 호스트 numpy 를 거친다(작은 복사).
RTMW_MEAN = (123.675, 116.28, 103.53)  # rtmlib RTMPose 기본(BGR 입력에 채널 순서대로 적용)
RTMW_STD = (58.395, 57.12, 57.375)


def _torchvision_decode():
    try:
        from torchvision.io import ImageReadMode, decode_jpeg

        return decode_jpeg, ImageReadMode.RGB
    except Exception:  # noqa: BLE001 - torchvision 이 없거나 torch 와 판이 안 맞으면 PIL 로
        return None, None


def decode_jpegs(datas: list[bytes], device="cpu") -> list:
    """JPEG 여러 장 -> (3,H,W) uint8 RGB 텐서 목록. GPU 면 nvJPEG(torchvision, 묶음 한 번). 없으면 PIL 로 풀어 올린다.
    한 장이라도 깨졌으면 예외(엔진이 한 장씩 다시 돌려 그 프레임만 버린다)."""
    import torch

    decode, rgb = _torchvision_decode()
    if decode is not None:
        raw = [torch.frombuffer(bytearray(d), dtype=torch.uint8) for d in datas]
        if str(device).startswith("cuda"):
            return list(decode(raw, mode=rgb, device=device))
        return [decode(r, mode=rgb) for r in raw]
    import io

    from PIL import Image

    out = []
    for d in datas:
        arr = np.asarray(Image.open(io.BytesIO(d)).convert("RGB")).copy()
        out.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous().to(device))
    return out


def gpu_decode_available() -> bool:
    return _torchvision_decode()[0] is not None


def _chw_float(img, device):
    """numpy HxWx3 uint8 또는 (3,H,W) uint8 텐서 -> (3,H,W) float 0~1."""
    import torch

    if isinstance(img, torch.Tensor):
        return img.to(device).float().div_(255.0)
    return _to_chw(img, device)


def _bgr255(img, device):
    """(3,H,W) uint8 RGB 텐서 -> (1,3,H,W) float BGR 0~255."""
    return img.to(device).flip(0).float()[None]


def warp_crops(img, crops: list[tuple], out_wh: tuple[int, int] = (192, 256)):
    """rtmlib RTMPose.preprocess(bbox_xyxy2cs(padding 1.25) -> 가로세로 비 맞춤 -> cv2.warpAffine INTER_LINEAR(밖은 0) ->
    uint8 -> (x-mean)/std) 를 GPU 의 grid_sample 로. crops: [(center, scale, warp_mat 2x3)]. 결과 (N,3,H,W) float32(BGR)."""
    import torch
    import torch.nn.functional as F

    dev = img.device
    src = _bgr255(img, dev)
    _, _, H, W = src.shape
    w, h = out_wh
    ys, xs = torch.meshgrid(torch.arange(h, device=dev, dtype=torch.float32),
                            torch.arange(w, device=dev, dtype=torch.float32), indexing="ij")
    ones = torch.ones_like(xs)
    dst = torch.stack([xs, ys, ones], dim=-1)  # (h, w, 3)
    grids = []
    for _, _, mat in crops:
        full = np.vstack([np.asarray(mat, dtype=np.float64), [0.0, 0.0, 1.0]])
        inv = torch.tensor(np.linalg.inv(full)[:2], device=dev, dtype=torch.float32)  # 출력 좌표 -> 원본 좌표
        sxy = dst @ inv.T  # (h, w, 2) 원본 픽셀 좌표(픽셀 중심 = 정수)
        gx = (2 * sxy[..., 0] + 1) / W - 1
        gy = (2 * sxy[..., 1] + 1) / H - 1
        grids.append(torch.stack([gx, gy], dim=-1))
    grid = torch.stack(grids)  # (N, h, w, 2)
    out = F.grid_sample(src.expand(len(crops), -1, -1, -1), grid, mode="bilinear", padding_mode="zeros",
                        align_corners=False)
    out = out.round_().clamp_(0, 255)  # cv2 는 uint8 로 돌려준다
    mean = torch.tensor(RTMW_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(RTMW_STD, device=dev).view(1, 3, 1, 1)
    return (out - mean) / std


def crop_params(box, out_wh: tuple[int, int] = (192, 256)):
    """rtmlib 과 같은 (center, scale, warp_mat). CPU 계산(숫자 몇 개)."""
    from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs, get_warp_matrix

    center, scale = bbox_xyxy2cs(np.array(box), padding=1.25)
    w, h = out_wh
    aspect = w / h
    b_w, b_h = np.hsplit(scale, [1])
    scale = np.where(b_w > b_h * aspect, np.hstack([b_w, b_w / aspect]), np.hstack([b_h * aspect, b_h]))
    mat = get_warp_matrix(center, scale, 0, output_size=(w, h))
    return center, scale, mat


def letterbox(img, size: tuple[int, int] = (416, 416)):
    """rtmlib YOLOX.preprocess(비율 유지 cv2.resize INTER_LINEAR -> uint8 -> 114 로 채움) 를 GPU 로. 결과 (1,3,H,W) float BGR, 비율."""
    import torch
    import torch.nn.functional as F

    src = _bgr255(img, img.device)
    _, _, H, W = src.shape
    if (H, W) == tuple(size):
        return src, 1.0
    ratio = min(size[0] / H, size[1] / W)
    rh, rw = int(H * ratio), int(W * ratio)
    resized = F.interpolate(src, size=(rh, rw), mode="bilinear", align_corners=False).round_().clamp_(0, 255)
    out = torch.full((1, 3, size[0], size[1]), 114.0, device=img.device)
    out[:, :, :rh, :rw] = resized
    return out, ratio
