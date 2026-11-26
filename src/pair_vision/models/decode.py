"""JPEG 디코드. 기본은 PIL 그대로, `fast=True` 면 더 빠른 길을 쓴다.

- PIL `draft`: JPEG 는 DCT 단계에서 1/2·1/4·1/8 로 줄여 풀 수 있어, 모델 입력(640~960)보다 훨씬 큰 사진만 크게 아낀다.
  결과 좌표는 비율(0~1)이라 크기가 줄어도 규칙에는 영향이 없다.
- turbojpeg(선택 설치 `pip install PyTurboJPEG`, 시스템에 libturbojpeg 필요): draft 로 못 줄이는 사진을 PIL 보다 빠르게 푼다.
"""

from __future__ import annotations

import io

_TURBO = None
_TURBO_TRIED = False
DEFAULT_TARGET_EDGE = 1280  # 이 긴 변보다 작아지지 않는 선까지만 줄인다(모델 입력 960 보다 큼)


def _turbo():
    global _TURBO, _TURBO_TRIED
    if not _TURBO_TRIED:
        _TURBO_TRIED = True
        try:
            from turbojpeg import TurboJPEG

            _TURBO = TurboJPEG()
        except Exception:  # noqa: BLE001 - 설치·시스템 라이브러리 없음: PIL 로
            _TURBO = None
    return _TURBO


def decode_jpeg(jpeg: bytes, *, fast: bool = False, target_edge: int = DEFAULT_TARGET_EDGE):
    """JPEG(또는 PNG 등 PIL 이 읽는 형식) 바이트를 RGB PIL 이미지로 푼다."""
    from PIL import Image

    image = Image.open(io.BytesIO(jpeg))
    if not fast:
        return image.convert("RGB")
    if image.format == "JPEG":
        long_edge = max(image.size)
        if long_edge >= 2 * target_edge:
            scale = target_edge / long_edge
            image.draft("RGB", (max(1, int(image.width * scale)), max(1, int(image.height * scale))))
        else:
            turbo = _turbo()
            if turbo is not None:
                try:
                    from turbojpeg import TJPF_RGB

                    return Image.fromarray(turbo.decode(jpeg, pixel_format=TJPF_RGB))
                except Exception:  # noqa: BLE001 - 이 사진만 PIL 로
                    pass
    return image.convert("RGB")
