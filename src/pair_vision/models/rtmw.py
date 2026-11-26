"""RTMW(rtmlib + ONNX Runtime) 백엔드: 손·얼굴·몸을 GPU 에서 한 모델로 분석한다 (MediaPipe 는 파이썬에서 CPU 로만 돌아 병목이었다).

- 휴대폰 프레임: 프레임 전체를 사람 한 명의 상자로 넣어 손 두 개(왼·오른)를 얻는다(사람 검출기를 거치면 팔이 여러 "사람"으로 잡혔다).
- 웹캠 프레임: 사람 검출기(YOLOX-tiny)로 사람 상자를 찾고(없으면 프레임 전체) 얼굴 수·머리 각도·어깨를 구한다.
onnxruntime·rtmlib 는 load() 안에서만 import 한다. CUDA 를 못 쓰면 CPU 로 돌고 `gpu` 가 False 로 보고된다.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from pair_vision.models.base import HandDet, WebcamFeatures
from pair_vision.models.postprocess import RTMW_DET_FILE, RTMW_POSE_FILE
from pair_vision.models.rtmw_features import hands_from_keypoints, webcam_from_keypoints
from pair_vision.rules.config import ModelRules

log = logging.getLogger("pair_vision.rtmw")

DET_FILE = RTMW_DET_FILE
POSE_FILE = RTMW_POSE_FILE
DET_INPUT = (416, 416)
POSE_INPUT = (192, 256)
DET_SCORE = 0.4


class RtmwBackend:
    def __init__(self, models_dir: str | Path, rules: ModelRules) -> None:
        self._dir = Path(models_dir)
        self._rules = rules
        self.gpu = False

    def load(self) -> None:
        import onnxruntime as ort

        try:  # onnxruntime>=1.21: torch 가 pip 로 깐 CUDA·cuDNN 라이브러리를 미리 올려 GPU 를 잡게 한다
            ort.preload_dlls()
        except Exception:  # noqa: BLE001 - 없는 버전이거나 라이브러리가 없으면 그냥 진행(아래에서 실제 provider 를 확인)
            pass
        from rtmlib import YOLOX, RTMPose

        device = "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
        self._det = YOLOX(str(self._dir / DET_FILE), model_input_size=DET_INPUT, score_thr=DET_SCORE,
                          backend="onnxruntime", device=device)
        self._pose = RTMPose(str(self._dir / POSE_FILE), model_input_size=POSE_INPUT, to_openpose=False,
                             backend="onnxruntime", device=device)
        used = self._pose.session.get_providers()
        self.gpu = "CUDAExecutionProvider" in used
        if device == "cuda" and not self.gpu:
            log.warning("CUDA 를 요청했지만 ONNX Runtime 이 CPU 로 돌아갑니다 (providers=%s)", used)
        if device == "cpu":
            log.warning("ONNX Runtime 에 CUDA provider 가 없어 CPU 로 돌립니다 (onnxruntime-gpu 설치 필요)")
        log.info("RTMW 적재 완료 (providers=%s)", used)

    @staticmethod
    def _bgr(rgb: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(rgb[:, :, ::-1])

    def hands(self, rgb: np.ndarray) -> list[HandDet]:
        """휴대폰 프레임의 손(점수 높은 순)."""
        height, width = rgb.shape[:2]
        box = np.array([[0, 0, width, height]], dtype=np.float32)
        kps, scores = self._pose(self._bgr(rgb), bboxes=box)
        return hands_from_keypoints(kps[0], scores[0], width, height, self._rules.hand_threshold)

    def webcam(self, rgb: np.ndarray) -> WebcamFeatures:
        """웹캠 프레임의 얼굴 수·머리 각도·어깨."""
        height, width = rgb.shape[:2]
        bgr = self._bgr(rgb)
        boxes = self._det(bgr)
        if len(boxes) == 0:  # 사람을 못 찾아도(머리만 크게 나온 경우 등) 프레임 전체로 얼굴 유무를 본다
            boxes = np.array([[0, 0, width, height]], dtype=np.float32)
        kps, scores = self._pose(bgr, bboxes=boxes)
        r = self._rules
        return webcam_from_keypoints(
            list(kps), list(scores), width, height,
            face_threshold=r.face_threshold, yaw_scale=r.yaw_scale, pitch_scale=r.pitch_scale,
        )
