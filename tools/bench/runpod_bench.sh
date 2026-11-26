#!/usr/bin/env bash
# RunPod GPU Pod 에서 모델 비교 측정을 한 번에 돌린다.
#   zip 으로 올린 경우:  cd <압축 푼 폴더> && bash tools/bench/runpod_bench.sh
#   Docker 이미지인 경우: bash /opt/pair-bench/tools/bench/runpod_bench.sh   (패키지·모델이 이미 들어 있어 설치를 건너뛴다)
# 필요한 것: CUDA 가 켜진 PyTorch Pod. 사진은 PHOTOS(기본 photos/ 또는 /workspace/photos)에 있으면 쓰고,
#            없으면 속도 측정용 합성 사진을 만든다(속도는 사진 내용과 거의 무관하다).
# 하위 명령(첫 인자):
#   (없음)            모델 비교 측정(속도 ms/장)  -> result_<GPU>.md/.json
#   capacity          용량 측정(가상 응시자 N명, 조합 workers x 배치 x fp16 x 웹캠 x 인원) -> capacity_<GPU>.md/.csv
#                     환경 변수로 일부만 고른다: WORKERS=1,2,4 BATCH=off,on FP16=off,on WEBCAM=server,off EXAMINEES=10,20,30 DURATION=20 WARMUP=5
#                     (모델 파일 전부 필요: dfine-small-coco owlv2-base-patch16-ensemble hand_landmarker.task face_landmarker.task pose_landmarker_lite.task)
# 결과: result_<GPU이름>.md(표) 와 result_<GPU이름>.json — /workspace 가 있으면 거기에(Jupyter 에서 바로 내려받기), 없으면 현재 폴더
set -euo pipefail
MODE="${1:-compare}"
cd "$(dirname "$0")/../.."

MODELS="${MODELS:-mediapipe-hands,dfine-s,rtdetr-r18,owlv2}"
WARMUP="${WARMUP:-5}"
REPEAT="${REPEAT:-10}"
if [ -z "${OUT_DIR:-}" ]; then
  if [ -d /workspace ]; then OUT_DIR=/workspace; else OUT_DIR=.; fi
fi

echo "== 1. GPU 확인"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'PY'
import sys

import torch

print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda 사용 가능:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("CUDA 를 쓸 수 없습니다. PyTorch+CUDA 템플릿인지, Pod 의 CUDA 버전이 이미지와 맞는지 확인하세요. 이 상태로 재면 CPU 값이 됩니다.")
    sys.exit(1)
PY

echo "== 2. 패키지 (torch 는 건드리지 않는다)"
if python -c "import transformers, scipy, PIL, numpy, huggingface_hub" 2>/dev/null; then
  echo "필요한 패키지가 이미 있음 -> 설치 건너뜀"
else
  pip install -q transformers scipy pillow numpy huggingface_hub
fi
python -c "import mediapipe" 2>/dev/null || pip install -q mediapipe || echo "[주의] mediapipe 설치 실패 -> 손 검출은 건너뜁니다"

echo "== 3. 모델 (이미 있으면 건너뜀)"
python - <<'PY'
from pathlib import Path

from huggingface_hub import snapshot_download

MODELS = {
    "rtdetr_r18vd_coco_o365": "PekingU/rtdetr_r18vd_coco_o365",
    "dfine-small-coco": "ustc-community/dfine-small-coco",
    "owlv2-base-patch16-ensemble": "google/owlv2-base-patch16-ensemble",
}
for folder, repo in MODELS.items():
    target = Path("models") / folder
    if (target / "config.json").is_file():
        print("있음:", folder)
        continue
    print("받는 중:", repo)
    snapshot_download(repo, local_dir=target, ignore_patterns=["*.bin", "*.onnx", "*.msgpack", "*.h5", "*.ot"])
PY
test -f models/hand_landmarker.task || echo "[주의] models/hand_landmarker.task 가 없습니다 -> 손 검출은 건너뜁니다"

echo "== 4. 사진"
PHOTOS="${PHOTOS:-}"
if [ -z "$PHOTOS" ]; then
  for candidate in photos /workspace/photos; do
    if ls "$candidate"/*.png "$candidate"/*.jpg "$candidate"/*.jpeg >/dev/null 2>&1; then PHOTOS="$candidate"; break; fi
  done
fi
if [ -z "$PHOTOS" ]; then
  PHOTOS=/tmp/synthetic-photos
  echo "사진 폴더가 없어 속도 측정용 합성 사진 8장을 만듭니다(검출 결과는 의미 없음, 속도만 본다)"
  python - "$PHOTOS" <<'PY'
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
for i in range(8):
    image = Image.fromarray((rng.random((380, 480, 3)) * 60 + 120).astype("uint8"))
    draw = ImageDraw.Draw(image)
    for _ in range(6):
        x0, y0 = int(rng.integers(0, 380)), int(rng.integers(0, 280))
        w, h = int(rng.integers(40, 100)), int(rng.integers(30, 80))
        draw.rectangle([x0, y0, x0 + w, y0 + h], fill=tuple(int(v) for v in rng.integers(30, 220, 3)))
    image.save(out / f"{i + 1}.png")
PY
fi
echo "사진 폴더: $PHOTOS"

if [ "$MODE" = "capacity" ]; then
  echo "== 5. 용량 측정 (조합이 많으면 1~2시간. WORKERS= EXAMINEES= 등으로 줄이세요)"
  GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' /' '__')"
  python -c "import pynvml" 2>/dev/null || pip install -q nvidia-ml-py psutil || true
  python -c "import psutil" 2>/dev/null || pip install -q psutil || true
  PYTHONPATH="src:." python -m tools.bench.capacity --detector real --no-verify     --photos "$PHOTOS" --models-dir models     --workers "${WORKERS:-1,2,4,8}" --batch "${BATCH:-off,on}" --fp16 "${FP16:-off,on}"     --webcam "${WEBCAM:-server,off}" --examinees "${EXAMINEES:-10,20,30,40}"     --duration "${DURATION:-20}" --warmup "${WARMUP:-5}"     --out "$OUT_DIR/capacity_${GPU_NAME}"
  echo
  echo "끝. 내려받을 파일: $OUT_DIR/capacity_${GPU_NAME}.md, $OUT_DIR/capacity_${GPU_NAME}.csv"
  exit 0
fi

echo "== 5. 측정"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' /' '__')"
PYTHONPATH="src:." python -m tools.bench.compare \
  --images "$PHOTOS" --models "$MODELS" \
  --warmup "$WARMUP" --repeat "$REPEAT" \
  --json "$OUT_DIR/result_${GPU_NAME}.json" | tee "$OUT_DIR/result_${GPU_NAME}.md"

echo
echo "끝. 내려받을 파일: $OUT_DIR/result_${GPU_NAME}.md, $OUT_DIR/result_${GPU_NAME}.json"
echo "(손 검출 mediapipe-hands 는 GPU 가 아니라 이 Pod 의 CPU 로 돈다)"
