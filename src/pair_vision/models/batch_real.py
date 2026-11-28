"""실제 모델 묶음 추론 백엔드(VISION_BATCH=1, VISION_DETECTOR=real): D-FINE·RTMW·OWLv2 를 응시자 사이 묶음으로 돌린다.

단계(엔진이 단계마다 대기 작업을 묶는다)
- 휴대폰: [dfine](세션마다 dfine_interval_ms 주기) -> rtmw(프레임 전체 한 상자 -> 손) -> [owlv2](owl_interval_ms 주기)
- 웹캠  : [person](YOLOX, person_box_interval_ms 주기, 사이에는 직전 상자 재사용) -> rtmw(사람 상자마다)
- rtmw 는 휴대폰 손과 웹캠 얼굴·몸을 **한 묶음**에 섞어 돌린다(같은 모델).
CPU 풀(prepare): JPEG 풀기·BGR 변환·RTMW 자르기(약 0.5ms)·주기 판단. GPU 이면 D-FINE·OWLv2 전처리는 엔진 스레드에서 GPU 로
(models.fast_preprocess), CPU 이면 전처리 스레드에서 torch 로 한다(OWLv2 기준 전처리 120ms/장 -> 수 ms).
주기(cadence)는 그 단계가 성공했을 때만 갱신한다(실패하면 다음 프레임에 다시). 칸마다 처리 중 1장이라 같은 세션 상태를
두 스레드가 동시에 고치지 않는다.
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path

from pair_vision.models.base import PHONE, Detection
from pair_vision.models.batch import BatchBackend, Job, pad_bucket
from pair_vision.models.postprocess import best_per_label
from pair_vision.models.real import RealDetector
from pair_vision.rules.config import ModelRules

log = logging.getLogger("pair_vision.batch_real")

GPU_STAGE_ORDER = ("dfine", "person", "rtmw", "owlv2")


class RealBatchBackend(RealDetector, BatchBackend):
    name = "real-batch"

    def __init__(self, models_dir: str | Path, rules: ModelRules, *, lock_path=None, verify: bool = True,
                 owl_max_batch: int = 4, warmup: bool | None = None, fp16_skip: frozenset[str] = frozenset(),
                 device_preprocess: bool | None = None) -> None:
        if rules.vision_backend != "rtmw":
            raise ValueError("묶음 추론(VISION_BATCH=1)은 models.vision_backend: rtmw 에서만 쓴다")
        super().__init__(models_dir, rules, lock_path=lock_path, verify=verify, fp16_skip=fp16_skip)
        self._owl_max = max(1, owl_max_batch)
        self._warmup = warmup  # None 이면 GPU 일 때만
        # 전부 GPU 경로(JPEG 풀기·D-FINE/OWLv2 전처리·RTMW 자르기·YOLOX 레터박스). None 이면 GPU + fast_preprocess 일 때.
        # True 면 CPU 에서도 같은 torch 경로를 탄다(시험용).
        self._device_pre_opt = device_preprocess
        self.device_preprocess = False

    # --- 적재 ---
    def load(self) -> None:
        super().load()
        opt = self._device_pre_opt
        self.device_preprocess = (self._rules.fast_preprocess and self._device == "cuda") if opt is None else bool(opt)
        if self.device_preprocess:
            from pair_vision.models.preprocess import gpu_decode_available

            log.info("전처리 전부 %s (JPEG 풀기: %s)", self._device,
                     "torchvision nvJPEG" if gpu_decode_available() else "PIL 후 GPU 로 올림(torchvision 없음)")
        if self._warmup if self._warmup is not None else self._device == "cuda":
            self._warm()

    def _warm(self) -> None:
        """GPU 첫 실행 지연(커널 적재·cuDNN 알고리즘 탐색)을 시험 전에 치른다. 묶음 크기마다 한 번씩."""
        import numpy as np

        t = time.perf_counter()
        rgb = np.full((384, 512, 3), 128, dtype=np.uint8)
        bgr = rgb[:, :, ::-1].copy()
        for n in range(1, 17):  # 묶음 크기마다(입력 모양이 처음 나올 때의 지연을 시험 전에 치른다)
            self.coco_batch([self.coco_pixels(rgb, self._device)] * n, [(384, 512)] * n)
        for n in range(1, self._owl_max + 1):
            self.owl_batch([self.owl_pixels(rgb, self._device)] * n, [(384, 512)] * n)
        crop = self._rtmw.crop(bgr, self._rtmw.full_frame(512, 384)[0])
        for n in (1, 2, 4, 8, 16, 32):
            self._rtmw.pose_batch([crop] * n)
        self._rtmw.person_boxes(bgr)
        if self.device_preprocess:
            import io

            from PIL import Image

            from pair_vision.models.preprocess import decode_jpegs

            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, "JPEG", quality=70)
            imgs = decode_jpegs([buf.getvalue()] * 4, self._device)
            self._rtmw.person_boxes_on_device(imgs[0])
            self._rtmw.crops_on_device(imgs[0], self._rtmw.full_frame(512, 384))
        log.info("GPU 예열 끝 (%.1f초)", time.perf_counter() - t)

    def max_batch(self, stage: str, default: int) -> int:
        return min(default, self._owl_max) if stage == "owlv2" else default

    def stats(self) -> dict[str, tuple[float, float]]:
        return self._timer.snapshot()  # decode(전처리 스레드). 단계별 시간은 엔진이 센다

    # --- CPU 풀 ---
    def prepare(self, job: Job) -> list[str]:
        if self.device_preprocess:
            return self._plan(job)
        import numpy as np
        from PIL import Image

        f = job.frame
        t = time.perf_counter()
        rgb = np.asarray(Image.open(io.BytesIO(f.jpeg)).convert("RGB"))
        self._timed("decode", t)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        height, width = rgb.shape[:2]
        st = job.state
        st.update(rgb=rgb, bgr=bgr, size=(height, width))
        cpu_pixels = self._device != "cuda" or not self._rules.fast_preprocess  # GPU+빠른 전처리면 엔진 스레드에서 GPU 로
        stages: list[str] = []
        if f.channel == PHONE:
            if self._coco_cadence.due(f.key, f.recv_ms) or f.key not in self._last_coco:
                stages.append("dfine")
                if cpu_pixels:
                    st["dfine_px"] = self.coco_pixels(rgb)
            st["crops"] = [self._rtmw.crop(bgr, self._rtmw.full_frame(width, height)[0])]
            stages.append("rtmw")
            if self._owl_cadence.due(f.key, f.recv_ms):
                stages.append("owlv2")
                if cpu_pixels:
                    st["owl_px"] = self.owl_pixels(rgb)
            return stages
        boxes = self._rtmw.boxes.get(f.key, f.recv_ms)
        if boxes is None:
            stages.append("person")
        else:
            st["crops"] = self._crops(bgr, boxes)
        stages.append("rtmw")
        return stages

    def _plan(self, job: Job) -> list[str]:
        """전부 GPU 경로: CPU 에서는 단계만 정한다(주기 판단). 풀기·전처리는 엔진 스레드에서 GPU 로."""
        f = job.frame
        stages = ["decode"]
        if f.channel == PHONE:
            if self._coco_cadence.due(f.key, f.recv_ms) or f.key not in self._last_coco:
                stages.append("dfine")
            stages.append("rtmw")
            if self._owl_cadence.due(f.key, f.recv_ms):
                stages.append("owlv2")
            job.state["boxes"] = None  # 프레임 전체(크기는 풀고 나서)
            return stages
        boxes = self._rtmw.boxes.get(f.key, f.recv_ms)
        if boxes is None:
            stages.append("person")
        job.state["boxes"] = boxes
        return stages + ["rtmw"]

    def _crops(self, bgr, boxes) -> list[tuple]:
        if len(boxes) == 0:  # 사람을 못 찾아도 프레임 전체로 얼굴 유무를 본다
            boxes = self._rtmw.full_frame(bgr.shape[1], bgr.shape[0])
        return [self._rtmw.crop(bgr, box) for box in boxes]

    # --- 엔진 스레드 ---
    def run_stage(self, stage: str, jobs: list[Job]) -> None:
        if self.device_preprocess and stage in ("decode", "person", "rtmw"):
            self._run_device_stage(stage, jobs)
        elif stage == "dfine":
            pixels = [j.state.pop("dfine_px", None) for j in jobs]
            pixels = [p if p is not None else self.coco_pixels(j.state["rgb"], self._device) for p, j in zip(pixels, jobs)]
            results = self.coco_batch(pixels, [j.state["size"] for j in jobs])
            for job, found in zip(jobs, results, strict=True):
                self._last_coco[job.frame.key] = best_per_label(found)
                self._coco_cadence.mark(job.frame.key, job.frame.recv_ms)
        elif stage == "owlv2":
            pixels = [j.state.pop("owl_px", None) for j in jobs]
            pixels = [p if p is not None else self.owl_pixels(j.state["rgb"], self._device) for p, j in zip(pixels, jobs)]
            results = self.owl_batch(pixels, [j.state["size"] for j in jobs])
            for job, found in zip(jobs, results, strict=True):
                job.state["owl"] = found
                self._owl_cadence.mark(job.frame.key, job.frame.recv_ms)
        elif stage == "person":
            for job in jobs:  # YOLOX ONNX 는 배치 1 고정
                boxes = self._rtmw.person_boxes(job.state["bgr"])
                self._rtmw.boxes.put(job.frame.key, job.frame.recv_ms, boxes)
                job.state["crops"] = self._crops(job.state["bgr"], boxes)
        elif stage == "rtmw":
            crops = [c for j in jobs for c in j.state["crops"]]
            poses = self._rtmw.pose_batch(crops)
            i = 0
            for job in jobs:
                n = len(job.state["crops"])
                job.state["poses"] = poses[i : i + n]
                i += n
        else:
            raise ValueError(f"모르는 단계: {stage}")

    def _run_device_stage(self, stage: str, jobs: list[Job]) -> None:
        if stage == "decode":  # nvJPEG 묶음 한 번
            from pair_vision.models.preprocess import decode_jpegs

            imgs = decode_jpegs([j.frame.jpeg for j in jobs], self._device)  # 시간은 엔진이 단계 'decode' 로 센다
            for job, img in zip(jobs, imgs, strict=True):
                job.state["rgb"] = img  # (3,H,W) uint8 텐서 - D-FINE·OWLv2 전처리 함수가 그대로 받는다
                job.state["size"] = (int(img.shape[1]), int(img.shape[2]))
        elif stage == "person":
            for job in jobs:
                boxes = self._rtmw.person_boxes_on_device(job.state["rgb"])
                self._rtmw.boxes.put(job.frame.key, job.frame.recv_ms, boxes)
                job.state["boxes"] = boxes
        else:  # rtmw: 응시자들의 상자를 GPU 에서 잘라 한 묶음으로
            for job in jobs:
                boxes = job.state["boxes"]
                img = job.state["rgb"]
                if boxes is None or len(boxes) == 0:
                    boxes = self._rtmw.full_frame(int(img.shape[2]), int(img.shape[1]))
                job.state["crops"] = self._rtmw.crops_on_device(img, boxes)
            crops = [c for j in jobs for c in j.state["crops"]]
            poses = self._rtmw.pose_batch(crops)
            i = 0
            for job in jobs:
                n = len(job.state["crops"])
                job.state["poses"] = poses[i : i + n]
                i += n

    def finish(self, job: Job) -> Detection:
        f, st = job.frame, job.state
        height, width = st["size"]
        if f.channel != PHONE:
            return Detection(key=f.key, channel=f.channel, seq=f.seq, recv_ms=f.recv_ms,
                             webcam=self._rtmw.webcam_from(st["poses"], width, height))
        hands = self._rtmw.hands_from(st["poses"][0], width, height)
        best = self._last_coco.get(f.key, {})  # 처리 중에 세션이 끝나(forget) 지워졌을 수 있다
        owl = st.get("owl", [])
        extra = [o for o in (*best.values(), *best_per_label(owl).values()) if o.label not in ("keyboard", "mouse")]
        return Detection(
            key=f.key, channel=f.channel, seq=f.seq, recv_ms=f.recv_ms,
            keyboard=best["keyboard"].box if "keyboard" in best else None,
            mouse=best["mouse"].box if "mouse" in best else None,
            hands=tuple(hands[:2]), objects=tuple(extra), hand_count=len(hands),
        )


__all__ = ["RealBatchBackend", "GPU_STAGE_ORDER", "pad_bucket"]
