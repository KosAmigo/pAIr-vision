"""C: 사진 1장당 CPU 쪽 단계 비용(ms) - JPEG 풀기, 전처리, 규칙 판정. 합성 JPEG(계약 크기)를 쓴다. 모델 추론은 재지 않는다.

실행(.venv-bench: PIL·numpy·transformers 필요): python tools/capacity_sim/cpu_stage_cost.py [--models models]
전처리(D-FINE·OWLv2 이미지 프로세서)는 가중치 폴더에 설정 파일이 있을 때만 잰다(없으면 건너뜀).
"""

from __future__ import annotations

import argparse
import io
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def synth(w: int, h: int, quality: int, seed: int) -> bytes:
    """사진처럼 저주파 + 약한 질감을 가진 합성 이미지 -> JPEG."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (h // 16 + 1, w // 16 + 1, 3), dtype=np.uint8)
    img = Image.fromarray(base).resize((w, h), Image.BICUBIC).filter(ImageFilter.GaussianBlur(2))
    arr = np.asarray(img).astype(np.int16) + rng.integers(-6, 7, (h, w, 3))
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def bench(fn, n: int = 200, warm: int = 10) -> tuple[float, float]:
    for _ in range(warm):
        fn()
    xs = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t) * 1000)
    xs.sort()
    return statistics.mean(xs), xs[int(len(xs) * 0.95)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    args = ap.parse_args()
    cases = {"웹캠 400x300 q55": (400, 300, 55), "웹캠 400x225(16:9) q55": (400, 225, 55), "휴대폰 512x384 q70": (512, 384, 70), "휴대폰 512x384 q85": (512, 384, 85)}
    print("| 입력 | JPEG 크기 | PIL 풀기+RGB(ms) 평균/p95 | numpy 변환+BGR 복사(ms) | 합계 |")
    print("|---|---|---|---|---|")
    jpegs = {}
    for name, (w, h, q) in cases.items():
        jpegs[name] = data = synth(w, h, q, 1)
        dec = bench(lambda d=data: Image.open(io.BytesIO(d)).convert("RGB"))
        img = Image.open(io.BytesIO(data)).convert("RGB")
        conv = bench(lambda i=img: np.ascontiguousarray(np.asarray(i)[:, :, ::-1]))
        print(f"| {name} | {len(data)/1024:.1f}KB | {dec[0]:.2f} / {dec[1]:.2f} | {conv[0]:.3f} | {dec[0]+conv[0]:.2f} |")

    # 규칙 판정: 가짜 Detection 으로 Session.on_detection(손 규칙·물체 규칙·웹캠 규칙)
    from pair_vision.models.base import PHONE, WEBCAM, Detection
    from pair_vision.models.fake import KEYBOARD, MOUSE, STATES, WEBCAM_STATES
    from pair_vision.rules.config import load_rules
    from pair_vision.session.session import Session

    rules = load_rules("configs/rules.default.yaml")
    profile = rules.profiles["default"] if hasattr(rules, "profiles") else rules.profile("default")
    s = Session("k" * 22, "exam", 0, profile)
    t = [0]
    hands, _ = STATES["NORMAL"]

    def phone() -> None:
        t[0] += 1000
        s.on_detection(Detection("k", PHONE, 1, t[0], keyboard=KEYBOARD, mouse=MOUSE, hands=hands))

    def cam() -> None:
        t[0] += 1000
        s.on_detection(Detection("k", WEBCAM, 1, t[0], webcam=WEBCAM_STATES["FRONT"]))

    for name, fn in (("휴대폰 규칙(손+물체)", phone), ("웹캠 규칙", cam)):
        a, p95 = bench(fn, 2000, 50)
        print(f"\n규칙 판정 {name}: 평균 {a*1000:.1f}us / p95 {p95*1000:.1f}us")

    # 모델 전처리(이미지 프로세서) - 가중치 설정 파일이 있을 때만
    root = Path(args.models)
    for sub, label, key in (("dfine-small-coco", "D-FINE 전처리", "휴대폰 512x384 q70"), ("owlv2-base-patch16-ensemble", "OWLv2 전처리(960 패딩·리사이즈)", "휴대폰 512x384 q70")):
        if not (root / sub).exists():
            print(f"\n{label}: 건너뜀({root/sub} 없음)")
            continue
        try:
            from transformers import AutoImageProcessor, AutoProcessor

            proc = (AutoProcessor if "owl" in sub else AutoImageProcessor).from_pretrained(root / sub)
            img = Image.open(io.BytesIO(jpegs[key])).convert("RGB")
            if "owl" in sub:
                a = bench(lambda: proc(text=[["a tablet", "a smartwatch on a wrist", "a smartphone"]], images=img, return_tensors="pt"), 20, 3)
            else:
                a = bench(lambda: proc(images=img, return_tensors="pt"), 50, 3)
            print(f"\n{label}: 평균 {a[0]:.1f}ms / p95 {a[1]:.1f}ms ({key})")
        except Exception as exc:  # noqa: BLE001
            print(f"\n{label}: 실패 {exc!r}")

    # RTMW 전처리(rtmlib: 아핀 크롭, cv2)와 YOLOX 전처리 - cv2 가 있을 때만 근사
    try:
        import cv2

        img = np.asarray(Image.open(io.BytesIO(jpegs["휴대폰 512x384 q70"])).convert("RGB"))
        bgr = np.ascontiguousarray(img[:, :, ::-1])

        def rtmw_pre() -> None:
            m = np.float32([[0.375, 0, 0], [0, 0.667, 0]])  # 192x256 입력으로 아핀 크롭 근사
            out = cv2.warpAffine(bgr, m, (192, 256)).astype(np.float32)
            out = (out - np.array([123.675, 116.28, 103.53], np.float32)) / np.array([58.395, 57.12, 57.375], np.float32)
            out.transpose(2, 0, 1)

        a = bench(rtmw_pre, 300, 10)
        print(f"\nRTMW 전처리(cv2 아핀+정규화, 근사): 평균 {a[0]:.2f}ms")

        def yolox_pre() -> None:
            out = cv2.resize(bgr, (416, 416)).astype(np.float32)
            out.transpose(2, 0, 1)

        a = bench(yolox_pre, 300, 10)
        print(f"YOLOX 전처리(cv2 리사이즈, 근사): 평균 {a[0]:.2f}ms")
    except ImportError:
        print("\ncv2 없음: RTMW 전처리 근사 건너뜀")


if __name__ == "__main__":
    main()
