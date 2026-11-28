#!/usr/bin/env bash
# RunPod(또는 리눅스 GPU 머신)에서 서버를 돌리는 데 필요한 것을 준비한다. 몇 번을 실행해도 안전하고, 이미 되어 있으면 건너뛴다.
#   1) 시스템 라이브러리(apt): Pod 를 다시 켤 때마다 사라지므로 매번 확인해 빠진 것만 설치
#   2) 파이썬 가상환경 $WORK/venv-vision (영구 볼륨 /workspace 에 둔다 -> 다음 기동 때 재설치 없음). Pod 에 깔린 torch 는 그대로 쓴다
#   3) 모델 5개 내려받기·sha256 검증 ($VISION_MODELS_DIR, 기본 <저장소>/models; 영구 볼륨에 둔다)
# 사용: bash tools/runpod/bootstrap.sh        환경 변수: WORK(기본 /workspace) VISION_VENV VISION_MODELS_DIR PYTHON
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
pick_work() {  # 코드·가상환경·모델을 둘 폴더. /workspace 는 쓰지 않는다(권한 문제): 지정값 > 홈 > /tmp > 현재 폴더
  local c
  for c in "${WORK:-}" "${HOME:-/tmp}/pair-vision-work" /tmp/pair-vision-work "$PWD/pair-vision-work"; do
    [ -n "$c" ] || continue
    if mkdir -p "$c" 2>/dev/null && touch "$c/.write-test" 2>/dev/null; then rm -f "$c/.write-test"; echo "$c"; return 0; fi
  done
  echo "쓸 수 있는 폴더가 없습니다" >&2; return 1
}
WORK="$(pick_work)"
VENV="${VISION_VENV:-$WORK/venv-vision}"
MODELS="${VISION_MODELS_DIR:-$ROOT/models}"
PY="${PYTHON:-python3}"
echo "저장소: $ROOT | 영구 작업 폴더: $WORK | 가상환경: $VENV | 모델: $MODELS"

echo "[1/3] 시스템 라이브러리"
if command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" = "0" ]; then
  missing=()
  for p in libgl1 libegl1 libgles2 libglib2.0-0 libxcb1 libsm6 libxext6 libxrender1; do
    # libglib2.0-0 은 배포판에 따라 이름이 바뀌어(t64) dpkg 로는 못 찾는다 -> 라이브러리 존재로 확인
    if [ "$p" = "libglib2.0-0" ] && ldconfig -p 2>/dev/null | grep -q "libglib-2.0.so"; then continue; fi
    dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "  설치: ${missing[*]}"
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "${missing[@]}" \
      || echo "  경고: apt 설치 실패 - opencv·onnxruntime 이 libGL/libxcb 오류를 내면 직접 설치해야 한다"
  else
    echo "  모두 있음"
  fi
else
  echo "  apt 없음 또는 root 아님 - 건너뜀(필요한 라이브러리: libgl1 libegl1 libgles2 libglib2.0-0 libxcb1)"
fi

echo "[2/3] 파이썬 환경"
WANT="$({ cat "$ROOT/pyproject.toml"; cat "$ROOT/uv.lock" 2>/dev/null || true; } | sha256sum | cut -d' ' -f1)"
if [ -x "$VENV/bin/python" ] && [ "$(cat "$VENV/.stamp" 2>/dev/null || true)" = "$WANT" ]; then
  echo "  이미 준비됨 (재설치 없음)"
else
  echo "  가상환경 만들고 패키지 설치 (처음 한 번, 몇 분)"
  "$PY" -m venv --system-site-packages "$VENV"
  "$VENV/bin/python" -m pip install -q -U pip
  "$VENV/bin/python" -m pip install -q "$ROOT[real]" huggingface_hub
  echo "$WANT" > "$VENV/.stamp"
fi
# transformers 5 는 torch>=2.6 이 필요하다. RunPod PyTorch 템플릿(예: torch 2.4)이 더 낮으면 가상환경에만 torch 2.6 을 설치한다.
# 시스템 torch 의 CUDA 버전에 맞는 휠(cu124/cu126/cu128)을 쓴다 - 최신 CUDA 휠은 Pod 의 드라이버보다 높아 GPU 를 못 잡을 수 있다.
torch_ok() {
  # torch>=2.6 이고, 보이는 torchvision·torchaudio 가 이 torch 와 맞아 불러와져야 한다(시스템의 옛 판이면 transformers 가 깨진다)
  "$VENV/bin/python" -c "import sys, importlib.util, torch; v = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); [__import__(m) for m in ('torchvision', 'torchaudio') if importlib.util.find_spec(m)]; sys.exit(0 if v >= (2, 6) else 1)" 2>/dev/null
}
if ! torch_ok; then
  CU="$("$VENV/bin/python" -c "import torch; print((torch.version.cuda or '').replace('.', '')[:3])" 2>/dev/null || true)"
  case "$CU" in 124|126|128) ;; *) CU=124 ;; esac
  echo "  torch 가 2.6 미만 -> 가상환경에 torch 2.6 (cu$CU) 설치 (수 분)"
  # --ignore-installed: 시스템(--system-site-packages)의 CUDA 라이브러리(nvidia-*)를 재사용하면 torch 2.6 이 자기 폴더에서
  # libcudnn.so.9 를 못 찾아 import 가 실패한다(RunPod PyTorch 2.4 이미지에서 확인, 2026-11-27). torch 와 그 CUDA 라이브러리를
  # 가상환경에 같이 깐다. torchvision·torchaudio 도 맞춘다(시스템 것은 torch 2.4 용이라 transformers 가 불러오다
  # undefined symbol 로 깨진다 - 같은 이미지에서 torchaudio 로 확인).
  "$VENV/bin/python" -m pip install -q --ignore-installed "torch>=2.6,<2.7" "torchvision>=0.21,<0.22" "torchaudio>=2.6,<2.7" --index-url "https://download.pytorch.org/whl/cu$CU" || { echo "  torch 2.6 설치 실패 - 위 오류를 확인하세요" >&2; exit 1; }
  "$VENV/bin/python" -c "import torch, torchvision, torchaudio" || { echo "  torch 2.6 을 깔았지만 불러오지 못함 - 위 오류를 확인하세요" >&2; exit 1; }
fi
# JPEG 을 GPU(nvJPEG)로 풀려면 torch 와 판이 맞는 torchvision 이 필요하다(없으면 서버는 PIL 로 풀어 GPU 로 올린다 - 동작은 함)
if ! "$VENV/bin/python" -c "from torchvision.io import decode_jpeg" 2>/dev/null; then
  TV="$("$VENV/bin/python" -c "import torch; m = int(torch.__version__.split('.')[1]); print(f'0.{m + 15}.*')")"
  CU="$("$VENV/bin/python" -c "import torch; print((torch.version.cuda or '').replace('.', '')[:3])" 2>/dev/null || true)"
  echo "  torchvision 없음 -> torchvision==$TV 설치(JPEG GPU 풀기용)"
  if [ -n "$CU" ]; then
    "$VENV/bin/python" -m pip install -q --no-deps "torchvision==$TV" --index-url "https://download.pytorch.org/whl/cu$CU" \
      || echo "  경고: torchvision 설치 실패 - JPEG 은 CPU(PIL)로 풀고 GPU 로 올립니다" >&2
  else
    "$VENV/bin/python" -m pip install -q --no-deps "torchvision==$TV" \
      || echo "  경고: torchvision 설치 실패 - JPEG 은 CPU(PIL)로 풀고 GPU 로 올립니다" >&2
  fi
fi
# RTMW(손·얼굴·몸)는 ONNX Runtime 으로 GPU 에서 돈다: GPU 가 있으면 onnxruntime(CPU) 을 onnxruntime-gpu 로 바꾼다.
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! "$VENV/bin/python" -c "import sys, onnxruntime as o; sys.exit(0 if 'CUDAExecutionProvider' in o.get_available_providers() else 1)" 2>/dev/null; then
    echo "  onnxruntime -> onnxruntime-gpu 교체"
    "$VENV/bin/python" -m pip uninstall -y -q onnxruntime >/dev/null 2>&1 || true
    "$VENV/bin/python" -m pip install -q onnxruntime-gpu || echo "  경고: onnxruntime-gpu 설치 실패 - RTMW 가 CPU 로 돌아 느립니다" >&2
  fi
fi

"$VENV/bin/python" - <<'PYEOF'
import torch, transformers, fastapi, onnxruntime
from rtmlib import YOLOX, RTMPose  # noqa: F401  RTMW 추론 라이브러리
from transformers import AutoModelForObjectDetection, AutoModelForZeroShotObjectDetection  # torch 가 낮으면 여기서 실패한다
print(f"  torch {torch.__version__} | CUDA 사용 가능: {torch.cuda.is_available()}"
      + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else " (GPU 없음)"))
print(f"  transformers {transformers.__version__} | onnxruntime {onnxruntime.__version__} {onnxruntime.get_available_providers()} | fastapi {fastapi.__version__}")
PYEOF

echo "[3/3] 모델"
"$VENV/bin/python" "$ROOT/tools/runpod/setup_models.py" --models-dir "$MODELS"
echo "준비 끝. 서버 실행: bash tools/runpod/start.sh   (VISION_API_KEY 환경 변수 필요)"
