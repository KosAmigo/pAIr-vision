# 기본은 CPU 이미지(가짜 모델, 의존성 최소).
# 실제 모델 이미지(D-FINE-s + OWLv2 + MediaPipe): UV_EXTRAS=--extra real. 가중치는 이미지에 넣지 않는다(볼륨 /models).
#   docker build --build-arg UV_EXTRAS="--extra real" -t pair-vision:real .
# GPU(RunPod): 위에 BASE_IMAGE=<CUDA 런타임 + Python 3.12 이미지> 를 더 준다(Linux 용 PyPI torch 는 CUDA 빌드가 들어 있다).
#   실행: docker run --gpus all -e VISION_DETECTOR=real -e VISION_MODELS_DIR=/models -v <모델 폴더>:/models:ro ...
ARG BASE_IMAGE=python:3.12-slim
ARG UV_EXTRAS=""
FROM ${BASE_IMAGE}
ARG UV_EXTRAS

COPY --from=ghcr.io/astral-sh/uv:0.13.10 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project ${UV_EXTRAS}
COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev ${UV_EXTRAS}

RUN useradd --system --uid 10001 vision
USER vision

EXPOSE 8000
# 접근 로그는 __main__ 에서 끈다(경로에 visionKey 가 있다). 코어 덤프는 compose/RunPod 에서 ulimit 0.
CMD ["python", "-m", "pair_vision"]
