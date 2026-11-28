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
from pair_vision.models.postprocess import RTMW_DET_FILE, RTMW_POSE_FILE, Cadence
from pair_vision.models.rtmw_features import hands_from_keypoints, webcam_from_keypoints
from pair_vision.rules.config import ModelRules

log = logging.getLogger("pair_vision.rtmw")

DET_FILE = RTMW_DET_FILE
POSE_FILE = RTMW_POSE_FILE
DET_INPUT = (416, 416)
POSE_INPUT = (192, 256)
DET_SCORE = 0.4
POSE_BATCH_CAP = 32  # RTMW 한 번에 넣는 사람 상자 수 상한(2의 거듭제곱으로 채워 입력 모양을 6가지로 고정)


class PersonBoxCache:
    """웹캠 사람 상자(YOLOX) 재사용(`models.person_box_interval_ms`): 세션마다 주기마다만 새로 찾고 사이에는 직전 상자를 쓴다.
    응시자는 의자에 앉아 크게 움직이지 않는다. RTMW 전처리가 상자를 1.25배 넓혀 잘라 작은 움직임은 상자 안에 남는다.
    새 사람이 들어온 것(MULTI_FACE)은 다음 검출 때(최대 주기만큼 늦게) 잡힌다."""

    def __init__(self, interval_ms: int) -> None:
        self._cadence = Cadence(interval_ms)
        self._boxes: dict[str, np.ndarray] = {}

    def get(self, key: str, now_ms: int) -> np.ndarray | None:
        """재사용할 상자(검출 결과 그대로, 빈 배열이면 '사람 없음'). 새로 찾을 때가 되었으면 None."""
        if key not in self._boxes or self._cadence.due(key, now_ms):
            return None
        return self._boxes[key]

    def put(self, key: str, now_ms: int, boxes: np.ndarray) -> None:
        self._boxes[key] = boxes
        self._cadence.mark(key, now_ms)

    def forget(self, key: str) -> None:
        self._boxes.pop(key, None)
        self._cadence.forget(key)


class RtmwBackend:
    def __init__(self, models_dir: str | Path, rules: ModelRules) -> None:
        self._dir = Path(models_dir)
        self._rules = rules
        self.gpu = False
        self.boxes = PersonBoxCache(rules.person_box_interval_ms)

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

    # --- 묶음 추론용 조각 ---
    def person_boxes(self, bgr: np.ndarray) -> np.ndarray:
        """YOLOX-tiny 사람 상자(N x 4, 픽셀 xyxy). ONNX 입력이 배치 1 고정이라 한 장씩 돈다."""
        boxes = np.asarray(self._det(bgr), dtype=np.float32)
        return boxes.reshape(-1, 4) if boxes.size else np.zeros((0, 4), dtype=np.float32)

    @staticmethod
    def full_frame(width: int, height: int) -> np.ndarray:
        return np.array([[0, 0, width, height]], dtype=np.float32)

    def crop(self, bgr: np.ndarray, box) -> tuple:
        """RTMW 입력 하나(정규화된 HxWx3, 중심, 크기). CPU 약 0.5ms."""
        return self._pose.preprocess(bgr, box)

    def pose_batch(self, crops: list[tuple]) -> list[tuple[np.ndarray, np.ndarray]]:
        """여러 상자(여러 응시자)를 RTMW 한 번(배치)에 돌린다. 결과: 상자마다 (점 133x2, 점수 133)."""
        from pair_vision.models.batch import pad_bucket

        out: list[tuple[np.ndarray, np.ndarray]] = []
        session = self._pose.session
        name = session.get_inputs()[0].name
        outputs = [o.name for o in session.get_outputs()]
        for start in range(0, len(crops), POSE_BATCH_CAP):
            chunk = crops[start : start + POSE_BATCH_CAP]
            # 자른 입력: CPU 경로는 HxWx3, GPU 경로(crops_on_device)는 이미 3xHxW
            x = np.stack([c[0] if c[0].shape[0] == 3 else c[0].transpose(2, 0, 1) for c in chunk]).astype(np.float32)
            size = pad_bucket(len(chunk), POSE_BATCH_CAP)
            if size > len(chunk):
                x = np.concatenate([x, np.zeros((size - len(chunk), *x.shape[1:]), dtype=np.float32)])
            res = session.run(outputs, {name: np.ascontiguousarray(x)})
            for i, (_, center, scale) in enumerate(chunk):
                kps, scores = self._pose.postprocess([r[i : i + 1] for r in res], center, scale)
                out.append((kps[0], scores[0]))
        return out

    # --- 전부 GPU 경로: 이미지는 (3,H,W) uint8 RGB 텐서 ---
    def person_boxes_on_device(self, img) -> np.ndarray:
        """YOLOX 레터박스를 GPU 에서 만들고 ONNX Runtime 으로 사람 상자를 찾는다(후처리는 rtmlib 그대로)."""
        from pair_vision.models.preprocess import letterbox

        x, ratio = letterbox(img, DET_INPUT)
        session = self._det.session
        out = session.run([o.name for o in session.get_outputs()],
                          {session.get_inputs()[0].name: np.ascontiguousarray(x.cpu().numpy(), dtype=np.float32)})
        boxes = np.asarray(self._det.postprocess(out[0], ratio), dtype=np.float32)
        return boxes.reshape(-1, 4) if boxes.size else np.zeros((0, 4), dtype=np.float32)

    def crops_on_device(self, img, boxes) -> list[tuple]:
        """사람 상자마다 RTMW 입력을 GPU 의 grid_sample 로 자른다(rtmlib 아핀과 같은 행렬). 결과는 pose_batch 입력."""
        from pair_vision.models.preprocess import crop_params, warp_crops

        if len(boxes) == 0:
            boxes = self.full_frame(int(img.shape[2]), int(img.shape[1]))
        params = [crop_params(box, POSE_INPUT) for box in boxes]
        tensors = warp_crops(img, params, POSE_INPUT).cpu().numpy()
        return [(tensors[i], c, s) for i, (c, s, _) in enumerate(params)]

    def hands_from(self, pose: tuple[np.ndarray, np.ndarray], width: int, height: int) -> list[HandDet]:
        return hands_from_keypoints(pose[0], pose[1], width, height, self._rules.hand_threshold)

    def webcam_from(self, poses: list[tuple[np.ndarray, np.ndarray]], width: int, height: int) -> WebcamFeatures:
        r = self._rules
        return webcam_from_keypoints(
            [p[0] for p in poses], [p[1] for p in poses], width, height,
            face_threshold=r.face_threshold, yaw_scale=r.yaw_scale, pitch_scale=r.pitch_scale,
        )

    def webcam_boxes(self, bgr: np.ndarray, key: str | None, now_ms: int | None) -> np.ndarray:
        """웹캠 사람 상자: 재사용 주기 안이면 직전 상자, 아니면 새로 검출. 사람이 없으면 프레임 전체(머리만 크게 나온 경우 등)."""
        boxes = self.boxes.get(key, now_ms) if key is not None and now_ms is not None else None
        if boxes is None:
            boxes = self.person_boxes(bgr)
            if key is not None and now_ms is not None:
                self.boxes.put(key, now_ms, boxes)
        if len(boxes) == 0:
            return self.full_frame(bgr.shape[1], bgr.shape[0])
        return boxes

    # --- 한 장씩(단일 경로) ---
    def hands(self, rgb: np.ndarray) -> list[HandDet]:
        """휴대폰 프레임의 손(점수 높은 순)."""
        height, width = rgb.shape[:2]
        bgr = self._bgr(rgb)
        pose = self.pose_batch([self.crop(bgr, self.full_frame(width, height)[0])])[0]
        return self.hands_from(pose, width, height)

    def webcam(self, rgb: np.ndarray, key: str | None = None, now_ms: int | None = None) -> WebcamFeatures:
        """웹캠 프레임의 얼굴 수·머리 각도·어깨."""
        height, width = rgb.shape[:2]
        bgr = self._bgr(rgb)
        boxes = self.webcam_boxes(bgr, key, now_ms)
        poses = self.pose_batch([self.crop(bgr, box) for box in boxes])
        return self.webcam_from(poses, width, height)
