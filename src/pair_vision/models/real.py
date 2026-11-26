"""실제 모델 Detector: D-FINE-s + OWLv2 + MediaPipe Hands (2026-11-26 확정, 그림자 모드 후 재확정).

- D-FINE-s: 휴대폰 프레임마다 키보드·마우스·노트(COCO book)
- OWLv2   : 세션마다 `owl_interval_ms` 주기로 태블릿·스마트워치·(다른)휴대폰
- MediaPipe Hands: 휴대폰 프레임마다 손(최대 `max_hands`), 가까운 중복은 합치고 점수 상위 2개만 규칙에 넘긴다
- 웹캠 채널(마일스톤 B): MediaPipe 얼굴 랜드마커(최대 3얼굴·블렌드셰이프·변환 행렬) + 포즈 랜드마커(lite)로
  프론트 pcRules 와 같은 특징값(WebcamFeatures)을 만든다. 규칙 판정은 session/rules 쪽이 한다.
torch·transformers·mediapipe·Pillow 는 선택 의존성이라 load() 안에서만 import 한다.
프레임 하나가 실패하면 그 프레임만 건너뛴다. 실패를 "손·물체 없음"으로 바꾸지 않는다(규칙이 거짓으로 쌓이는 것 방지).

실행 옵션(환경 변수, models/options.py): `VISION_BATCH_GPU`(프레임 묶음 추론) `VISION_FP16` `VISION_FAST_DECODE` `VISION_WEBCAM=off`.
기본은 모두 꺼져 있어 기존 동작(프레임 한 장씩)과 같다.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from pathlib import Path

from pair_vision.models.base import (
    PHONE,
    Box,
    Detection,
    Detector,
    Frame,
    HandDet,
    ObjectDet,
    ScanDecodeError,
)
from pair_vision.models.decode import decode_jpeg
from pair_vision.models.lock import load_lock, verify_lock
from pair_vision.models.options import RunOptions
from pair_vision.models.postprocess import (
    COCO_LABELS,
    OWL_PROMPTS,
    SCAN_COCO_LABELS,
    SCAN_FIXED_THRESHOLDS,
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
    SCAN_CHUNK = 8  # 방 스캔에서 한 번에 모델에 넣는 사진 수 상한(GPU 메모리)

    def __init__(
        self,
        models_dir: str | Path,
        rules: ModelRules,
        *,
        lock_path: str | Path | None = None,
        verify: bool = True,
        options: RunOptions | None = None,
        threads: int = 0,
    ) -> None:
        self._threads = threads  # 0 이면 실효 CPU(cgroup 한도 반영) 만큼
        self._dir = Path(models_dir)
        self._rules = rules
        self._lock_path = lock_path
        self._verify = verify
        self._opt = options or RunOptions.from_env()
        self._half = False
        self._stats: dict[str, list[float]] = {}  # 단계 -> [프레임 수, 총 ms]
        self._stats_lock = threading.Lock()
        self._last_owl_ms: dict[str, int] = {}
        self._webcam_frames: dict[str, int] = {}  # 웹캠 프레임 수(포즈를 N 프레임마다만 돌리려고)
        self._last_pose: dict[str, object] = {}  # 직전 포즈 랜드마크(사이 프레임에서 재사용)
        self.gpu = False
        self.threads = 0

    # --- 적재 ---
    def load(self) -> None:
        for sub in (DFINE_DIR, OWL_DIR, HAND_FILE, FACE_FILE, POSE_FILE):
            if not (self._dir / sub).exists():
                raise FileNotFoundError(f"모델 파일 없음: {self._dir / sub} (VISION_MODELS_DIR 확인)")
        if self._verify and self._lock_path:
            verify_lock(self._dir, load_lock(self._lock_path))
        import mediapipe as mp
        import torch
        from mediapipe.tasks.python import BaseOptions, vision
        from transformers import (
            AutoImageProcessor,
            AutoModelForObjectDetection,
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        self._torch, self._mp = torch, mp
        from pair_vision.models.cpuinfo import resolve_threads

        self.threads = resolve_threads(self._threads, 1)  # 워커 1개(풀 없음)일 때도 실효 CPU 기준으로 스레드를 정한다
        torch.set_num_threads(self.threads)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu = self._device == "cuda"
        self._dfine_proc = AutoImageProcessor.from_pretrained(self._dir / DFINE_DIR)
        self._dfine = AutoModelForObjectDetection.from_pretrained(self._dir / DFINE_DIR).to(self._device).eval()
        self._owl_proc = AutoProcessor.from_pretrained(self._dir / OWL_DIR)
        self._owl = (
            AutoModelForZeroShotObjectDetection.from_pretrained(self._dir / OWL_DIR).to(self._device).eval()
        )
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
        self._thresholds = {
            "keyboard": r.keyboard_threshold, "mouse": r.mouse_threshold, "note": r.note_threshold,
            "tablet": r.tablet_threshold, "smartwatch": r.smartwatch_threshold, "phone": r.phone_threshold,
        }
        if self._opt.fp16 and self.gpu:  # cuda 에서만 half (CPU 에서는 무시)
            self._dfine.half()
            self._owl.half()
            self._half = True
        log.info("모델 적재 완료 (device=%s, fp16=%s, %s)", self._device, self._half, self._opt.label())

    # --- 단계별 시간(측정 도구용) ---
    def _tick(self, name: str, n: int, started: float) -> None:
        ms = (time.perf_counter() - started) * 1000
        with self._stats_lock:
            acc = self._stats.setdefault(name, [0, 0.0])
            acc[0] += n
            acc[1] += ms

    def stage_stats(self, reset: bool = False) -> dict[str, tuple[int, float]]:
        with self._stats_lock:
            snap = {k: (int(v[0]), v[1]) for k, v in self._stats.items()}
            if reset:
                self._stats.clear()
        return snap

    def forget(self, key: str) -> None:
        self._last_owl_ms.pop(key, None)
        self._webcam_frames.pop(key, None)
        self._last_pose.pop(key, None)

    # --- 실행 ---
    def run(self, frames: Sequence[Frame]) -> list[Detection]:
        if not self._opt.webcam:  # VISION_WEBCAM=off: 웹캠 프레임은 분석하지 않고 결과도 내지 않는다
            frames = [f for f in frames if f.channel == PHONE]
        if self._opt.batch_gpu:
            return self._run_batched(frames)
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

    def _decode(self, jpeg: bytes):
        started = time.perf_counter()
        image = decode_jpeg(jpeg, fast=self._opt.fast_decode)
        self._tick("decode", 1, started)
        return image

    def _run_batched(self, frames: Sequence[Frame]) -> list[Detection]:
        """VISION_BATCH_GPU=1: 휴대폰 프레임은 한 번에 D-FINE(주기가 된 것만 OWLv2)에 묶어 넣고, 손·웹캠은 프레임별(CPU)."""
        slots: list[Detection | None] = [None] * len(frames)
        phone: list[tuple[int, Frame, object]] = []
        failures = 0
        for i, frame in enumerate(frames):
            try:
                image = self._decode(frame.jpeg)
                if frame.channel == PHONE:
                    phone.append((i, frame, image))
                else:
                    slots[i] = self._webcam_result(frame, image)
            except Exception:  # noqa: BLE001
                failures += 1
                log.exception("프레임 분석 실패 (건너뜀)")
        if phone:
            try:
                dets = self._phone_batch([(f, im) for _, f, im in phone])
                for (i, _, _), det in zip(phone, dets, strict=True):
                    slots[i] = det
            except Exception:  # noqa: BLE001 - 묶음이 실패하면 한 장씩 다시(깨진 한 장이 전체를 막지 않게)
                log.exception("휴대폰 묶음 분석 실패 -> 한 장씩 다시")
                for i, frame, image in phone:
                    try:
                        slots[i] = self._phone_result(frame, image)
                    except Exception:  # noqa: BLE001
                        failures += 1
                        log.exception("프레임 분석 실패 (건너뜀)")
        results = [d for d in slots if d is not None]
        if failures and not results:
            raise RuntimeError("배치의 모든 프레임 분석 실패")
        return results

    def _phone_batch(self, items: list[tuple[Frame, object]]) -> list[Detection]:
        images = [im for _, im in items]
        cocos = self._detect_coco_batch(images)
        due = [k for k, (f, _) in enumerate(items) if self._owl_due(f)]
        owls: dict[int, list[ObjectDet]] = {}
        if due:
            for k, found in zip(due, self._detect_owl_batch([images[k] for k in due]), strict=True):
                owls[k] = found
        out = []
        for k, (frame, image) in enumerate(items):
            hands = self._timed_hands(image)
            if k in owls:
                self._last_owl_ms[frame.key] = frame.recv_ms  # 성공했을 때만 갱신
            out.append(self._phone_detection(frame, cocos[k], hands, owls.get(k, [])))
        return out

    def _owl_due(self, frame: Frame) -> bool:
        last = self._last_owl_ms.get(frame.key)
        return last is None or frame.recv_ms - last >= self._rules.owl_interval_ms

    def _run_phone(self, frame: Frame) -> Detection:
        return self._phone_result(frame, self._decode(frame.jpeg))

    def _phone_result(self, frame: Frame, image) -> Detection:
        coco = self._detect_coco_batch([image])[0]
        hands = self._timed_hands(image)
        owl: list[ObjectDet] = []
        if self._owl_due(frame):
            owl = self._detect_owl_batch([image])[0]
            self._last_owl_ms[frame.key] = frame.recv_ms  # 성공했을 때만 갱신(실패하면 다음 프레임에 다시)
        return self._phone_detection(frame, coco, hands, owl)

    def _phone_detection(
        self, frame: Frame, coco: list[ObjectDet], hands: list[HandDet], owl: list[ObjectDet]
    ) -> Detection:
        best = best_per_label(coco)
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
        return self._webcam_result(frame, self._decode(frame.jpeg))

    def _webcam_result(self, frame: Frame, image) -> Detection:
        import numpy as np

        mp_image = self._mp.Image(self._mp.ImageFormat.SRGB, np.asarray(image))
        started = time.perf_counter()
        face = self._face.detect(mp_image)
        self._tick("face", 1, started)
        count_before = self._webcam_frames.get(frame.key, 0)
        self._webcam_frames[frame.key] = count_before + 1
        if count_before % self._rules.pose_every == 0 or frame.key not in self._last_pose:
            started = time.perf_counter()
            pose = self._pose.detect(mp_image)  # 몸통은 천천히 움직이므로 포즈는 드물게(프론트도 얼굴 3 : 포즈 1)
            self._tick("pose", 1, started)
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

    # --- 모델 호출(사진 여러 장을 한 번에 넣을 수 있다) ---
    def _to_device(self, inputs):
        torch = self._torch
        inputs = inputs.to(self._device)
        if self._half:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(torch.float16)
        return inputs

    def _to_float(self, out):
        if self._half:  # 후처리(시그모이드·박스 변환)는 fp32 로
            for k, v in list(out.items()):
                if self._torch.is_tensor(v) and v.is_floating_point():
                    out[k] = v.float()
        return out

    def _dfine_raw(self, images: list, floor: float) -> list[list[tuple[str, float, Box]]]:
        """D-FINE 한 번 호출로 사진 여러 장의 (COCO 이름, 점수, 박스) 목록을 낸다."""
        torch = self._torch
        started = time.perf_counter()
        inputs = self._to_device(self._dfine_proc(images=images, return_tensors="pt"))
        with torch.inference_mode():
            out = self._to_float(self._dfine(**inputs))
        results = self._dfine_proc.post_process_object_detection(
            out, threshold=floor, target_sizes=[(im.height, im.width) for im in images]
        )
        id2label = self._dfine.config.id2label
        raw = []
        for image, res in zip(images, results, strict=True):
            raw.append([
                (id2label[int(label_id)], float(score),
                 Box(box[0] / image.width, box[1] / image.height, box[2] / image.width, box[3] / image.height))
                for score, label_id, box in zip(res["scores"], res["labels"], res["boxes"].tolist(), strict=True)
            ])
        self._tick("dfine", len(images), started)
        return raw

    def _detect_coco_batch(self, images: list) -> list[list[ObjectDet]]:
        floor = min(self._thresholds[k] for k in ("keyboard", "mouse", "note"))
        found = []
        for dets in self._dfine_raw(images, floor):
            kept = []
            for name, score, box in dets:
                label = COCO_LABELS.get(name)
                if label is not None and score >= self._thresholds[label]:
                    kept.append(ObjectDet(label, score, box))
            found.append(kept)
        return found

    def _owl_raw(self, images: list, prompts: list[str], floor: float) -> list[list[tuple[int, float, Box]]]:
        torch = self._torch
        started = time.perf_counter()
        inputs = self._to_device(
            self._owl_proc(text=[list(prompts)] * len(images), images=images, return_tensors="pt")
        )
        with torch.inference_mode():
            out = self._to_float(self._owl(**inputs))
        post = (
            getattr(self._owl_proc, "post_process_grounded_object_detection", None)
            or self._owl_proc.post_process_object_detection
        )
        sizes = [(max(im.height, im.width),) * 2 for im in images]  # OWLv2 는 긴 변 기준 정사각형 평면 좌표를 낸다
        results = post(out, threshold=floor, target_sizes=sizes)
        raw = []
        for image, res in zip(images, results, strict=True):
            raw.append([
                (int(label_id), float(score), box_in_image(box, image.width, image.height))
                for score, label_id, box in zip(res["scores"], res["labels"], res["boxes"].tolist(), strict=True)
            ])
        self._tick("owl", len(images), started)
        return raw

    def _detect_owl_batch(self, images: list) -> list[list[ObjectDet]]:
        prompts = list(OWL_PROMPTS)
        floor = min(self._thresholds[k] for k in ("tablet", "smartwatch", "phone"))
        found = []
        for dets in self._owl_raw(images, prompts, floor):
            kept = []
            for idx, score, box in dets:
                label = OWL_PROMPTS[prompts[idx]]
                if score >= self._thresholds[label]:
                    kept.append(ObjectDet(label, score, box))
            found.append(kept)
        return found

    def _timed_hands(self, image) -> list[HandDet]:
        started = time.perf_counter()
        hands = self._detect_hands(image)
        self._tick("hands", 1, started)
        return hands

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

    # --- 방 스캔(POST /v1/scan) ---
    def scan(self, jpegs: Sequence[bytes]) -> list[list[ObjectDet]]:
        """사진마다 person·phone·tv·laptop·book(D-FINE)과 tablet(OWLv2 `a tablet`). 실시간과 같은 모델을 묶음으로 쓴다."""
        images = []
        for jpeg in jpegs:
            try:
                images.append(decode_jpeg(jpeg, fast=self._opt.fast_decode))
            except Exception as err:  # noqa: BLE001
                raise ScanDecodeError("cannot decode image") from err
        limits = {**SCAN_FIXED_THRESHOLDS, "phone": self._thresholds["phone"], "book": self._thresholds["note"]}
        tablet_floor = self._thresholds["tablet"]
        out: list[list[ObjectDet]] = []
        for start in range(0, len(images), self.SCAN_CHUNK):
            chunk = images[start : start + self.SCAN_CHUNK]
            coco = self._dfine_raw(chunk, min(limits.values()))
            owl = self._owl_raw(chunk, ["a tablet"], tablet_floor)
            for dets, tablets in zip(coco, owl, strict=True):
                kept = []
                for name, score, box in dets:
                    label = SCAN_COCO_LABELS.get(name)
                    if label is not None and score >= limits[label]:
                        kept.append(ObjectDet(label, score, box))
                kept += [ObjectDet("tablet", score, box) for _, score, box in tablets if score >= tablet_floor]
                out.append(kept)
        return out
