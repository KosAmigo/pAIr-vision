"""실행 옵션(환경 변수로 켠다). 기본값은 기존 동작과 같다.

| 환경 변수 | 기본 | 뜻 |
|---|---|---|
| `VISION_BATCH_GPU` | 0 | 1 이면 RealDetector.run 이 받은 프레임을 묶어 한 번에 모델에 넣는다(D-FINE·OWLv2) |
| `VISION_FP16` | 0 | 1 이면 cuda 에서 모델을 half 로 돌린다(CPU 에서는 무시) |
| `VISION_FAST_DECODE` | 0 | 1 이면 JPEG 를 빠른 길로 푼다(PIL draft 로 줄여서 디코드, turbojpeg 가 있으면 사용) |
| `VISION_WEBCAM` | server | off 이면 웹캠 프레임을 서버에서 분석하지 않는다(용량 비교용) |
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


def _flag(env: Mapping[str, str], name: str) -> bool:
    return env.get(name, "0").strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class RunOptions:
    batch_gpu: bool = False
    fp16: bool = False
    fast_decode: bool = False
    webcam: bool = True  # False 면 웹캠 프레임은 분석하지 않고 결과도 내지 않는다

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> RunOptions:
        env = os.environ if env is None else env
        return cls(
            batch_gpu=_flag(env, "VISION_BATCH_GPU"),
            fp16=_flag(env, "VISION_FP16"),
            fast_decode=_flag(env, "VISION_FAST_DECODE"),
            webcam=env.get("VISION_WEBCAM", "server").strip().lower() not in ("off", "0", "false", "no"),
        )

    def label(self) -> str:
        return (
            f"batch={'on' if self.batch_gpu else 'off'} fp16={'on' if self.fp16 else 'off'} "
            f"webcam={'server' if self.webcam else 'off'}"
        )
