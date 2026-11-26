"""models/ 폴더를 읽어 configs/models.lock 을 만든다(이름·출처·라이선스는 아래 표에서, sha256 은 파일에서).

사용: python tools/models/make_lock.py [--models-dir models] [--out configs/models.lock]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pair_vision.models.lock import build_lock_entry  # noqa: E402

# 확정 모델(2026-11-26). 라이선스는 HF 모델 카드 기준(docs/model-candidates.md). 제외 모델은 넣지 않는다.
MODELS = [
    {
        "name": "dfine-s", "role": "키보드·마우스·노트(COCO book), 휴대폰 프레임마다",
        "repo": "ustc-community/dfine-small-coco", "license": "Apache-2.0",
        "directory": "dfine-small-coco",
    },
    {
        "name": "owlv2", "role": "태블릿·스마트워치·다른 휴대폰, 세션마다 몇 초 주기",
        "repo": "google/owlv2-base-patch16-ensemble", "license": "Apache-2.0",
        "directory": "owlv2-base-patch16-ensemble",
    },
    {
        "name": "rtmw-yolox-tiny", "role": "웹캠 사람 검출(RTMW 앞단)",
        "repo": "https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose",
        "license": "Apache-2.0 (OpenMMLab 코드·가중치 기준; 학습 데이터 조건은 확인 필요)",
        "directory": "rtmw/yolox_tiny.onnx",
        "zip_url": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
        "zip_member": "20230928/yolox_onnx/yolox_tiny_8xb8-300e_humanart-6f3252f9/end2end.onnx",
    },
    {
        "name": "rtmw-dw-l-m", "role": "손·얼굴 랜드마크·몸(어깨) 133점, GPU(ONNX Runtime) - MediaPipe 대체",
        "repo": "https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose",
        "license": "Apache-2.0 (OpenMMLab 코드·가중치 기준; COCO-WholeBody 등 학습 데이터 조건은 확인 필요)",
        "directory": "rtmw/rtmw-dw-l-m_256x192.onnx",
        "zip_url": "https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.zip",
        "zip_member": "end2end.onnx",
    },
    {
        "name": "mediapipe-hands", "role": "손 랜드마크",
        "repo": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "license": "Apache-2.0 (MediaPipe 코드 기준; 모델 파일 자체 조건은 미확인)",
        "directory": "hand_landmarker.task",
    },
    {
        "name": "mediapipe-face", "role": "웹캠 얼굴 랜드마크·블렌드셰이프·변환 행렬(마일스톤 B)",
        "repo": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "license": "Apache-2.0 (MediaPipe 코드 기준; 모델 파일 자체 조건은 미확인)",
        "directory": "face_landmarker.task",
    },
    {
        "name": "mediapipe-pose-lite", "role": "웹캠 포즈 랜드마크(어깨, 마일스톤 B)",
        "repo": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "license": "Apache-2.0 (MediaPipe 코드 기준; 모델 파일 자체 조건은 미확인)",
        "directory": "pose_landmarker_lite.task",
    },
]

HEADER = """# 확정 모델의 이름·출처·라이선스·파일 sha256 (설계 §15-2). 가중치는 저장소에 넣지 않는다.
# 서버는 시작할 때 models 폴더(VISION_MODELS_DIR)의 파일을 이 sha256 과 대조한다(VISION_VERIFY_MODELS=0 으로 끔).
# 갱신: python tools/models/make_lock.py
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", default="models")
    p.add_argument("--out", default="configs/models.lock")
    args = p.parse_args()
    entries = [build_lock_entry(Path(args.models_dir), m) for m in MODELS]
    Path(args.out).write_text(
        HEADER + yaml.safe_dump({"models": entries}, allow_unicode=True, sort_keys=False, width=200),
        encoding="utf-8",
    )
    print(f"{len(entries)}개 모델 -> {args.out}")


if __name__ == "__main__":
    main()
