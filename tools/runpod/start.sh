#!/usr/bin/env bash
# 준비(bootstrap.sh)를 확인한 뒤 서버를 실행한다. RunPod 템플릿의 "Container Start Command" 로 써도 된다:
#   bash /workspace/pAIr-vision/tools/runpod/start.sh
# 필요한 환경 변수: VISION_API_KEY (필수), VISION_CALLBACK_URL·VISION_CALLBACK_SECRET (백엔드로 이벤트를 보낼 때)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${WORK:-/workspace}"
[ -d "$WORK" ] || WORK="$HOME"
: "${VISION_API_KEY:?VISION_API_KEY 환경 변수가 필요합니다}"
bash "$ROOT/tools/runpod/bootstrap.sh"
cd "$ROOT"  # 규칙·lock 의 상대 경로(configs/...)가 저장소 기준이다
export PYTHONPATH="$ROOT/src"
export VISION_DETECTOR=real
export VISION_MODELS_DIR="${VISION_MODELS_DIR:-$ROOT/models}"
exec "${VISION_VENV:-$WORK/venv-vision}/bin/python" -m pair_vision
