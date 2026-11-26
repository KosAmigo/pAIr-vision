"""실제 모델 Detector: D-FINE-s + OWLv2 + MediaPipe Hands (2026-11-26 확정, 그림자 모드 후 재확정).

- D-FINE-s: 휴대폰 프레임마다 키보드·마우스·노트(COCO book)
- OWLv2   : 세션마다 `owl_interval_ms` 주기로 태블릿·스마트워치·(다른)휴대폰
- MediaPipe Hands: 휴대폰 프레임마다 손(최대 `max_hands`), 가까운 중복은 합치고 점수 상위 2개만 규칙에 넘긴다
- 웹캠 채널(마일스톤 B): MediaPipe 얼굴 랜드마커(최대 3얼굴·블렌드셰이프·변환 행렬) + 포즈 랜드마커(lite)로
  프론트 pcRules 와 같은 특징값(WebcamFeatures)을 만든다. 규칙 판정은 session/rules 쪽이 한다.
torch·transformers·mediapipe·Pillow 는 선택 의존성이라 load() 안에서만 import 한다.
프레임 하나가 실패하면 그 프레임만 건너뛴다. 실패를 "손·물체 없음"으로 바꾸지 않는다(규칙이 거짓으로 쌓이는 것 방지).
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from pair_vision.models.base import PHONE, Box, Detection, Detector, Frame, HandDet, ObjectDet
from pair_vision.models.lock import load_lock, verify_lock
from pair_vision.models.postprocess import (
    COCO_LABELS,
    Cadence,
    StageTimer,
    RTMW_DET_FILE,
    RTMW_POSE_FILE,
    OWL_PROMPTS,
    best_per_label,
    box_in_image,
    merge_close_hands,
)
from pair_vision.models.webcam_features import webcam_features
from pair_vision.rules.config import ModelRules

log = logging.getLogger("pair_vision.models")

DFINE_DIR = "dfine-small-coco"
OWL_DIR = "owlv2-base-patch16-ensemble"
HAND_FILE = "hand_landmarker.task"
FACE_FILE = "face_landmarker.task"
POSE_FILE = "pose_landmarker_lite.task"
WEBCAM_FACES = 3  # 프론트 numFaces 와 같다


class RealDetector(Detector):
    name = "real"

    def __init__(
        self,
        models_dir: str | Path,
        rules: ModelRules,
        *,
        lock_path: str | Path | None = None,
        verify: bool = True,
    ) -> None:
        self._dir = Path(models_dir)
        self._rules = rules
        self._lock_path = lock_path
        self._verify = verify
        self._owl_cadence = Cadence(rules.owl_interval_ms)
        self._coco_cadence = Cadence(rules.dfine_interval_ms)
        self._last_coco: dict[str, dict[str, ObjectDet]] = {}  # 세션별 직전 D-FINE 결과(주기 사이에 재사용)
        self._timer = StageTimer()
        self._webcam_frames: dict[str, int] = {}  # 웹캠 프레임 수(포즈를 N 프레임마다만 돌리려고)
        self._last_pose: dict[str, object] = {}  # 직전 포즈 랜드마크(사이 프레임에서 재사용)
        self.gpu = False

    # --- 적재 ---
    def load(self) -> None:
        rtmw = self._rules.vision_backend == "rtmw"
        needed = (DFINE_DIR, OWL_DIR) + ((RTMW_DET_FILE, RTMW_POSE_FILE) if rtmw else (HAND_FILE, FACE_FILE, POSE_FILE))
        for sub in needed:
            if not (self._dir / sub).exists():
                raise FileNotFoundError(f"모델 파일 없음: {self._dir / sub} (VISION_MODELS_DIR 확인)")
        if self._verify and self._lock_path:
            verify_lock(self._dir, load_lock(self._lock_path))
        import torch
        from transformers import (
            AutoImageProcessor,
            AutoModelForObjectDetection,
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu = self._device == "cuda"
        self._dfine_proc = AutoImageProcessor.from_pretrained(self._dir / DFINE_DIR)
        self._dfine = AutoModelForObjectDetection.from_pretrained(self._dir / DFINE_DIR).to(self._device).eval()
        self._owl_proc = AutoProcessor.from_pretrained(self._dir / OWL_DIR)
        self._owl = (
            AutoModelForZeroShotObjectDetection.from_pretrained(self._dir / OWL_DIR).to(self._device).eval()
        )
        r = self._rules
        self._rtmw = None
        if rtmw:  # 손·얼굴·몸을 GPU 한 모델(RTMW)로
            from pair_vision.models.rtmw import RtmwBackend  # onnxruntime·rtmlib 는 이 백엔드를 쓸 때만 필요

            self._rtmw = RtmwBackend(self._dir, r)
            self._rtmw.load()
            self.gpu = self.gpu and self._rtmw.gpu  # 하나라도 CPU 로 돌면 gpu=False 로 알린다
        else:
            self._load_mediapipe()
        self._owl_texts = [list(OWL_PROMPTS)]
        self._thresholds = {
            "keyboard": r.keyboard_threshold, "mouse": r.mouse_threshold, "note": r.note_threshold,
            "tablet": r.tablet_threshold, "smartwatch": r.smartwatch_threshold, "phone": r.phone_threshold,
        }
        log.info("모델 적재 완료 (device=%s, 비전 백엔드=%s, gpu=%s)", self._device, r.vision_backend, self.gpu)

    def _load_mediapipe(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        r = self._rules
        self._hands = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self._dir / HAND_FILE)),
                num_hands=r.max_hands,
                min_hand_detection_confidence=r.hand_threshold,
                min_hand_presence_confidence=r.hand_threshold,
                min_tracking_confidence=r.hand_threshold,
            )
        )
        self._face = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self._dir / FACE_FILE)),
                num_faces=WEBCAM_FACES,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
        )
        self._pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(self._dir / POSE_FILE)), num_poses=1
            )
        )

    def forget(self, key: str) -> None:
        self._owl_cadence.forget(key)
        self._coco_cadence.forget(key)
        self._last_coco.pop(key, None)
        self._webcam_frames.pop(key, None)
        self._last_pose.pop(key, None)

    # --- 실행 ---
    def run(self, frames: Sequence[Frame]) -> list[Detection]:
        results: list[Detection] = []
        failures = 0
        for frame in frames:
            try:
                results.append(self._run_phone(frame) if frame.channel == PHONE else self._run_webcam(frame))
            except Exception:  # noqa: BLE001 - 프레임 하나 때문에 배치 전체를 잃지 않는다
                failures += 1
                log.exception("프레임 분석 실패 (건너뜀)")
        if failures and not results:
            raise RuntimeError("배치의 모든 프레임 분석 실패")
        return results

    def stats(self) -> dict[str, tuple[float, float]]:
        return self._timer.snapshot()

    def _timed(self, stage: str, start: float) -> None:
        self._timer.add(stage, time.perf_counter() - start)

    def _run_phone(self, frame: Frame) -> Detection:
        import numpy as np
        from PIL import Image

        t = time.perf_counter()
        image = Image.open(io.BytesIO(frame.jpeg)).convert("RGB")
        self._timed("decode", t)
        if self._coco_cadence.due(frame.key, frame.recv_ms) or frame.key not in self._last_coco:
            t = time.perf_counter()
            self._last_coco[frame.key] = best_per_label(self._detect_coco(image))
            self._coco_cadence.mark(frame.key, frame.recv_ms)  # 성공했을 때만 갱신(실패하면 다음 프레임에 다시)
            self._timed("dfine", t)
        best = self._last_coco[frame.key]
        t = time.perf_counter()
        hands = self._rtmw.hands(np.asarray(image)) if self._rtmw is not None else self._detect_hands(image)
        self._timed("hands", t)
        owl: list[ObjectDet] = []
        if self._owl_cadence.due(frame.key, frame.recv_ms):
            t = time.perf_counter()
            owl = self._detect_owl(image)
            self._owl_cadence.mark(frame.key, frame.recv_ms)  # 성공했을 때만 갱신
            self._timed("owlv2", t)
        extra = [o for o in (*best.values(), *best_per_label(owl).values()) if o.label not in ("keyboard", "mouse")]
        return Detection(
            key=frame.key, channel=frame.channel, seq=frame.seq, recv_ms=frame.recv_ms,
            keyboard=best["keyboard"].box if "keyboard" in best else None,
            mouse=best["mouse"].box if "mouse" in best else None,
            hands=tuple(hands[:2]),
            objects=tuple(extra),
            hand_count=len(hands),
        )

    def _run_webcam(self, frame: Frame) -> Detection:
        import numpy as np
        from PIL import Image

        t = time.perf_counter()
        image = Image.open(io.BytesIO(frame.jpeg)).convert("RGB")
        self._timed("decode", t)
        if self._rtmw is not None:
            t = time.perf_counter()
            features = self._rtmw.webcam(np.asarray(image))
            self._timed("webcam", t)
            return Detection(key=frame.key, channel=frame.channel, seq=frame.seq, recv_ms=frame.recv_ms, webcam=features)
        mp_image = self._mp.Image(self._mp.ImageFormat.SRGB, np.asarray(image))
        face = self._face.detect(mp_image)
        count_before = self._webcam_frames.get(frame.key, 0)
        self._webcam_frames[frame.key] = count_before + 1
        if count_before % self._rules.pose_every == 0 or frame.key not in self._last_pose:
            pose = self._pose.detect(mp_image)  # 몸통은 천천히 움직이므로 포즈는 드물게(프론트도 얼굴 3 : 포즈 1)
            self._last_pose[frame.key] = pose.pose_landmarks[0] if pose.pose_landmarks else None
        count = len(face.face_landmarks)
        matrix = face.facial_transformation_matrixes[0] if count and face.facial_transformation_matrixes else None
        blend = (
            {c.category_name: c.score for c in face.face_blendshapes[0]}
            if count and face.face_blendshapes
            else None
        )
        landmarks = self._last_pose.get(frame.key)
        return Detection(
            key=frame.key, channel=frame.channel, seq=frame.seq, recv_ms=frame.recv_ms,
            webcam=webcam_features(count, matrix, blend, landmarks),
        )

    def _detect_coco(self, image) -> list[ObjectDet]:
        torch = self._torch
        inputs = self._dfine_proc(images=image, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            out = self._dfine(**inputs)
        floor = min(self._thresholds[k] for k in ("keyboard", "mouse", "note"))
        res = self._dfine_proc.post_process_object_detection(
            out, threshold=floor, target_sizes=[(image.height, image.width)]
        )[0]
        id2label = self._dfine.config.id2label
        found = []
        for score, label_id, box in zip(res["scores"], res["labels"], res["boxes"], strict=True):
            label = COCO_LABELS.get(id2label[int(label_id)])
            if label is None or float(score) < self._thresholds[label]:
                continue
            x0, y0, x1, y1 = box.tolist()
            found.append(
                ObjectDet(label, float(score), Box(x0 / image.width, y0 / image.height, x1 / image.width, y1 / image.height))
            )
        return found

    def _detect_owl(self, image) -> list[ObjectDet]:
        torch = self._torch
        inputs = self._owl_proc(text=self._owl_texts, images=image, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            out = self._owl(**inputs)
        side = max(image.height, image.width)  # OWLv2 는 긴 변 기준 정사각형 평면 좌표를 낸다
        post = (
            getattr(self._owl_proc, "post_process_grounded_object_detection", None)
            or self._owl_proc.post_process_object_detection
        )
        floor = min(self._thresholds[k] for k in ("tablet", "smartwatch", "phone"))
        res = post(out, threshold=floor, target_sizes=[(side, side)])[0]
        prompts = list(OWL_PROMPTS)
        found = []
        for score, label_id, box in zip(res["scores"], res["labels"], res["boxes"], strict=True):
            label = OWL_PROMPTS[prompts[int(label_id)]]
            if float(score) >= self._thresholds[label]:
                found.append(ObjectDet(label, float(score), box_in_image(box.tolist(), image.width, image.height)))
        return found

    def _detect_hands(self, image) -> list[HandDet]:
        import numpy as np

        mp_image = self._mp.Image(self._mp.ImageFormat.SRGB, np.asarray(image))
        result = self._hands.detect(mp_image)
        hands = []
        for lms, handed in zip(result.hand_landmarks, result.handedness, strict=True):
            x = (lms[0].x + lms[9].x) / 2  # 손목(0)과 손바닥 중심(9)의 중점
            y = (lms[0].y + lms[9].y) / 2
            hands.append(HandDet(handed[0].category_name.lower(), x, y, float(handed[0].score)))
        return merge_close_hands(hands, self._rules.hand_merge_distance)
