"""GPU 합격 시험용 합성 사진을 만든다(응시자 실사진 없음). 계약 크기·화질과 비슷하게:
휴대폰 512x384 JPEG q70(책상·키보드·마우스·손 모양), 웹캠 400x300 q55(얼굴·어깨 모양). 잡음을 섞어 실제 JPEG 크기·디코드 비용에 가깝게 한다.

  python tools/gpu_test/make_images.py --out <폴더> [--count 8]
결과: <폴더>/phone/*.jpg, <폴더>/webcam/*.jpg, <폴더>/probe/probe.jpg(손 없는 책상: 콜백·서명 확인용 이벤트). 모델 비용(배치·단계 시간)을 재는 것이 목적이라 검출 정확도와는 무관하다.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def _noise(img, rng: random.Random, amount: int):
    import numpy as np
    from PIL import Image

    arr = np.asarray(img).astype(np.int16)
    arr += np.random.default_rng(rng.randrange(1 << 30)).integers(-amount, amount + 1, arr.shape, dtype=np.int16)
    return Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))


def phone_image(i: int, rng: random.Random):
    from PIL import Image, ImageDraw, ImageFilter

    w, h = 512, 384
    img = Image.new("RGB", (w, h), (150 + rng.randrange(40), 120 + rng.randrange(30), 90 + rng.randrange(30)))
    d = ImageDraw.Draw(img)
    for y in range(0, h, 6):  # 나뭇결
        d.line((0, y + rng.randrange(4), w, y + rng.randrange(4)), fill=(130 + rng.randrange(40), 100, 70), width=1)
    kx, ky = 70 + rng.randrange(30), 190 + rng.randrange(30)
    d.rectangle((kx, ky, kx + 300, ky + 110), fill=(35, 35, 38))
    for r in range(5):  # 자판
        for c in range(14):
            x0, y0 = kx + 8 + c * 21, ky + 8 + r * 20
            d.rectangle((x0, y0, x0 + 16, y0 + 15), fill=(70, 70, 75))
    mx, my = kx + 340 + rng.randrange(20), ky + 40
    d.ellipse((mx, my, mx + 45, my + 70), fill=(220, 220, 225))
    skin = (215, 170 + rng.randrange(20), 140)
    for hx in (kx + 60 + rng.randrange(30), mx - 10):  # 손 두 개
        hy = ky + 30 + rng.randrange(30)
        d.ellipse((hx, hy, hx + 70, hy + 55), fill=skin)
        d.rectangle((hx + 15, hy + 40, hx + 55, h), fill=(60, 80, 140))  # 소매
    if i % 3 == 0:  # 몇 장에는 태블릿 같은 판
        d.rounded_rectangle((20, 30, 170, 140), radius=10, fill=(20, 20, 25))
        d.rectangle((30, 40, 160, 130), fill=(90, 120, 170))
    return _noise(img.filter(ImageFilter.GaussianBlur(0.6)), rng, 10)


def webcam_image(i: int, rng: random.Random):
    from PIL import Image, ImageDraw, ImageFilter

    w, h = 400, 300
    img = Image.new("RGB", (w, h), (180 + rng.randrange(40), 185, 190))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 120, 200), fill=(150, 160, 175))  # 뒤 벽·책장
    cx = 200 + rng.randrange(-25, 25)
    d.polygon([(cx - 130, h), (cx - 90, 210), (cx + 90, 210), (cx + 130, h)], fill=(50, 70, 110))  # 어깨
    d.rectangle((cx - 25, 170, cx + 25, 215), fill=(205, 160, 130))  # 목
    d.ellipse((cx - 55, 60, cx + 55, 190), fill=(215, 170, 140))  # 얼굴
    d.ellipse((cx - 60, 45, cx + 60, 110), fill=(40, 30, 25))  # 머리카락
    for ex in (cx - 25, cx + 25):
        d.ellipse((ex - 9, 115, ex + 9, 125), fill=(255, 255, 255))
        d.ellipse((ex - 4, 116, ex + 4, 124), fill=(30, 20, 15))
    d.line((cx, 125, cx - 4, 150), fill=(180, 130, 110), width=3)
    d.arc((cx - 20, 150, cx + 20, 170), 20, 160, fill=(150, 70, 70), width=3)
    return _noise(img.filter(ImageFilter.GaussianBlur(0.5)), rng, 16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    rng = random.Random(20261127)
    sizes = {}
    for kind, make, quality in (("phone", phone_image, 70), ("webcam", webcam_image, 55)):
        folder = out / kind
        folder.mkdir(parents=True, exist_ok=True)
        total = 0
        for i in range(args.count):
            path = folder / f"{i + 1:02d}.jpg"
            make(i, rng).save(path, "JPEG", quality=quality)
            total += path.stat().st_size
        sizes[kind] = total // args.count
    # 콜백 확인용: 손이 없는 책상(실제 모델 -> 손 이탈 HANDS_OUT), 가짜 모델용 표식 FAKE:BOTH_OUT 을 JPEG 주석에 넣는다
    from PIL import Image

    (out / "probe").mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (512, 384), (160, 130, 100)).save(out / "probe" / "probe.jpg", "JPEG", quality=70,
                                                     comment=b"FAKE:BOTH_OUT")
    print(f"합성 사진 {args.count}장씩 만듦: {out} (평균 크기 휴대폰 {sizes['phone'] // 1024}KB, 웹캠 {sizes['webcam'] // 1024}KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
