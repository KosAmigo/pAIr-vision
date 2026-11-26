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
      || echo "  경고: apt 설치 실패 - mediapipe 가 libEGL/libxcb 오류를 내면 직접 설치해야 한다"
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
  "$VENV/bin/python" -c "import sys, torch; v = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); sys.exit(0 if v >= (2, 6) else 1)" 2>/dev/null
}
if ! torch_ok; then
  CU="$("$VENV/bin/python" -c "import torch; print((torch.version.cuda or '').replace('.', '')[:3])" 2>/dev/null || true)"
  case "$CU" in 124|126|128) ;; *) CU=124 ;; esac
  echo "  torch 가 2.6 미만 -> 가상환경에 torch 2.6 (cu$CU) 설치 (수 분)"
  "$VENV/bin/python" -m pip install -q "torch>=2.6,<2.7" --index-url "https://download.pytorch.org/whl/cu$CU"     || { echo "  torch 2.6 설치 실패 - 위 오류를 확인하세요" >&2; exit 1; }
fi
"$VENV/bin/python" - <<'PYEOF'
import torch, transformers, mediapipe, fastapi
from transformers import AutoModelForObjectDetection, AutoModelForZeroShotObjectDetection  # torch 가 낮으면 여기서 실패한다
print(f"  torch {torch.__version__} | CUDA 사용 가능: {torch.cuda.is_available()}"
      + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else " (GPU 없음)"))
print(f"  transformers {transformers.__version__} | mediapipe {mediapipe.__version__} | fastapi {fastapi.__version__}")
PYEOF

echo "[3/3] 모델"
"$VENV/bin/python" "$ROOT/tools/runpod/setup_models.py" --models-dir "$MODELS"
echo "준비 끝. 서버 실행: bash tools/runpod/start.sh   (VISION_API_KEY 환경 변수 필요)"
