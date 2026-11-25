"""비교 대상 모델 어댑터. 무거운 라이브러리는 `load()` 안에서만 import 한다.

⚠ 실제 모델 어댑터(Hf*, MediaPipeHands)는 **아직 실제 가중치로 돌려 보지 못한 골격**이다
(허락 전이라 파일이 없다). 가중치를 받은 뒤 첫 실행에서 오류가 나면 이 파일을 고친다.
`fake` 어댑터만 지금 검증된다(측정 틀 자체의 시험용).
"""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from pathlib import Path

from pair_vision.models.base import Box, HandDet

from .harness import BenchModel, ObjectDet, RawResult

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# COCO 에서 시험 환경에 필요한 물체
WANTED_COCO = ("keyboard", "mouse", "cell phone", "laptop", "book", "remote")
# open-vocab 에 줄 문장 (설계 §8-1)
DEFAULT_PROMPTS = (
    "a keyboard",
    "a computer mouse",
    "a smartphone",
    "a tablet",
    "handwritten notes on paper",
    "a smartwatch on a wrist",
)


# 문장 -> 규칙 엔진·보고서가 쓰는 짧은 라벨
PROMPT_LABELS = {
    "a keyboard": "keyboard",
    "a computer mouse": "mouse",
    "a smartphone": "phone",
    "a tablet": "tablet",
    "handwritten notes on paper": "note",
    "a smartwatch on a wrist": "smartwatch",
}


def label_for_text(text: str, prompts: Sequence[str]) -> str:
    """모델이 돌려준 문구(Grounding DINO 는 조각난 문구를 낸다)를 가장 잘 맞는 프롬프트의 짧은 라벨로."""
    bare = text.lower().strip()
    best = ""
    for prompt in prompts:
        core = prompt.removeprefix("a ").removeprefix("an ")
        if (prompt == bare or core in bare) and len(core) > len(best):
            best = prompt
    return PROMPT_LABELS.get(best, text) if best else text


class FakeBench(BenchModel):
    """측정 틀 시험용. 사진을 읽지 않고 고정 결과를 낸다."""

    name = "fake"

    def infer(self, jpeg: bytes) -> RawResult:
        return RawResult(
            objects=(
                ObjectDet("keyboard", 0.9, Box(0.2, 0.5, 0.6, 0.9)),
                ObjectDet("mouse", 0.8, Box(0.65, 0.6, 0.8, 0.8)),
            ),
        )


def _open_rgb(jpeg: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(jpeg)).convert("RGB")


def _pick_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


# MediaPipe 손 검출의 기본 기준값(라이브러리 기본과 같다)
DEFAULT_HAND_CONF = 0.5


def _norm(box: Sequence[float], image) -> Box:
    x0, y0, x1, y1 = box
    return Box(x0 / image.width, y0 / image.height, x1 / image.width, y1 / image.height)


def box_in_image(box: Sequence[float], width: int, height: int) -> Box:
    """픽셀 박스를 사진 크기로 나눠 0~1 로 바꾼다(사진 밖으로 나간 부분은 잘라낸다).

    OWLv2 는 사진을 긴 변 기준 정사각형으로 패딩한 평면의 좌표를 내지만, 사진은 그 평면의 왼쪽 위에
    놓인다. 그러므로 사진의 가로·세로로 나누면 곧바로 사진 기준 좌표가 되고, 추가로 곱할 것이 없다
    (예전에는 긴 변/짧은 변 비율을 한 번 더 곱해 가로로 긴 사진에서 세로 위치가 1 을 넘었다).
    """
    x0, y0, x1, y1 = (min(1.0, max(0.0, v)) for v in (box[0] / width, box[1] / height, box[2] / width, box[3] / height))
    return Box(x0, y0, x1, y1)


def _require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"모델 폴더 없음: {path} (사용자 허락 후 내려받기)")


class HfClosedSet(BenchModel):
    """transformers 의 객체 검출 모델(RT-DETR, D-FINE, DETR). 가중치는 models/<폴더> 에 둔다."""

    def __init__(self, directory: str, label: str, threshold: float = 0.3) -> None:
        self.name = label
        self._dir = MODELS_DIR / directory
        self._threshold = threshold

    def load(self) -> None:
        _require_dir(self._dir)
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self._device = _pick_device()
        self.gpu = self._device == "cuda"
        self._proc = AutoImageProcessor.from_pretrained(self._dir)
        self._model = AutoModelForObjectDetection.from_pretrained(self._dir).to(self._device).eval()

    def infer(self, jpeg: bytes) -> RawResult:
        import torch

        image = _open_rgb(jpeg)
        inputs = self._proc(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model(**inputs)
        res = self._proc.post_process_object_detection(
            out, threshold=self._threshold, target_sizes=[(image.height, image.width)]
        )[0]
        id2label = self._model.config.id2label
        objects = []
        for score, label_id, box in zip(res["scores"], res["labels"], res["boxes"], strict=True):
            label = id2label[int(label_id)]
            if label in WANTED_COCO:
                objects.append(ObjectDet(label, float(score), _norm(box.tolist(), image)))
        return RawResult(objects=tuple(objects))


class HfZeroShot(BenchModel):
    """open-vocab: OWLv2 / Grounding DINO. 문장 목록을 그대로 라벨로 쓴다."""

    def __init__(
        self,
        directory: str,
        label: str,
        kind: str,
        prompts: Sequence[str] = DEFAULT_PROMPTS,
        threshold: float = 0.2,
    ) -> None:
        if kind not in ("owlv2", "grounding-dino"):
            raise ValueError(kind)
        self.name, self._dir, self._kind = label, MODELS_DIR / directory, kind
        self._prompts, self._threshold = tuple(prompts), threshold

    def load(self) -> None:
        _require_dir(self._dir)
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._device = _pick_device()
        self.gpu = self._device == "cuda"
        self._proc = AutoProcessor.from_pretrained(self._dir)
        self._model = (
            AutoModelForZeroShotObjectDetection.from_pretrained(self._dir).to(self._device).eval()
        )

    def infer(self, jpeg: bytes) -> RawResult:
        import torch

        image = _open_rgb(jpeg)
        if self._kind == "owlv2":
            inputs = self._proc(text=[list(self._prompts)], images=image, return_tensors="pt")
        else:  # grounding dino 는 소문자 문장을 ' . ' 로 이어 붙인다
            text = " . ".join(self._prompts) + " ."
            inputs = self._proc(text=text, images=image, return_tensors="pt")
        inputs = inputs.to(self._device)
        with torch.no_grad():
            out = self._model(**inputs)
        if self._kind == "owlv2":
            # OWLv2 는 긴 변 기준 정사각형 패딩 평면의 좌표를 낸다(사진은 그 왼쪽 위). 좌표를 사진 크기로 나누는 것은 box_in_image 가 한다
            side = max(image.height, image.width)
            post = getattr(self._proc, "post_process_grounded_object_detection", None) or self._proc.post_process_object_detection
            res = post(out, threshold=self._threshold, target_sizes=[(side, side)])[0]
            names = [label_for_text(self._prompts[int(i)], self._prompts) for i in res["labels"]]
        else:
            res = self._proc.post_process_grounded_object_detection(
                out,
                inputs.input_ids,
                threshold=self._threshold,
                text_threshold=self._threshold,
                target_sizes=[(image.height, image.width)],
            )[0]
            names = [label_for_text(str(n), self._prompts)
                     for n in res.get("text_labels", res.get("labels", []))]
        objects = [
            ObjectDet(str(name), float(score), box_in_image(box.tolist(), image.width, image.height))
            for score, name, box in zip(res["scores"], names, res["boxes"], strict=True)
        ]
        return RawResult(objects=tuple(objects))


class MediaPipeHands(BenchModel):
    """MediaPipe HandLandmarker(.task). 손 위치는 손목(랜드마크 0)과 손바닥 중심(9)의 중점."""

    name = "mediapipe-hands"

    def __init__(self, task_file: str = "hand_landmarker.task", conf: float = DEFAULT_HAND_CONF) -> None:
        if not 0.0 < conf <= 1.0:
            raise ValueError(f"손 감지 기준값은 0 초과 1 이하여야 합니다: {conf}")
        self._path = MODELS_DIR / task_file
        self._conf = conf
        # 기준값을 바꾼 것은 표에서 구분되도록 이름에 붙인다(예: mediapipe-hands@0.1)
        self.name = "mediapipe-hands" if conf == DEFAULT_HAND_CONF else f"mediapipe-hands@{conf:g}"

    def load(self) -> None:
        if not self._path.is_file():
            raise FileNotFoundError(f"모델 파일 없음: {self._path} (사용자 허락 후 내려받기)")
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        # 기준값을 낮추면 마우스·펜을 쥔 손처럼 잘 안 잡히던 손이 더 잡히지만 손이 아닌 것을 잡을 위험도 는다
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self._path)),
            num_hands=2,
            min_hand_detection_confidence=self._conf,
            min_hand_presence_confidence=self._conf,
            min_tracking_confidence=self._conf,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._mp = mp

    def infer(self, jpeg: bytes) -> RawResult:
        import numpy as np

        rgb = np.asarray(_open_rgb(jpeg))
        result = self._landmarker.detect(self._mp.Image(self._mp.ImageFormat.SRGB, rgb))
        hands = []
        for lms, handed in zip(result.hand_landmarks, result.handedness, strict=True):
            x = (lms[0].x + lms[9].x) / 2
            y = (lms[0].y + lms[9].y) / 2
            hands.append(HandDet(handed[0].category_name.lower(), x, y))
        return RawResult(hands=tuple(hands))


# 이름 -> 만드는 함수. 폴더 이름은 docs/model-candidates.md 의 "받을 위치" 와 같다.
REGISTRY: dict[str, Callable[[], BenchModel]] = {
    "fake": FakeBench,
    "rtdetr-r18": lambda: HfClosedSet("rtdetr_r18vd_coco_o365", "rtdetr-r18"),
    "rtdetr-r50": lambda: HfClosedSet("rtdetr_r50vd", "rtdetr-r50"),
    "dfine-s": lambda: HfClosedSet("dfine-small-coco", "dfine-s"),
    "owlv2": lambda: HfZeroShot("owlv2-base-patch16-ensemble", "owlv2", "owlv2"),
    "gdino-t": lambda: HfZeroShot("grounding-dino-tiny", "gdino-t", "grounding-dino"),
    "mediapipe-hands": MediaPipeHands,
}
