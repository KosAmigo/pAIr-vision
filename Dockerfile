# 기본은 CPU 이미지(가짜 모델). GPU 는 실제 모델·런타임(D-6)이 정해진 뒤 BASE_IMAGE 로 CUDA 이미지를 지정한다.
#   docker build --build-arg BASE_IMAGE=<CUDA 런타임 + Python 3.12 이미지> -t pair-vision:gpu .
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

COPY --from=ghcr.io/astral-sh/uv:0.13.10 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 PATH="/app/.venv/bin:$PATH"
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev

RUN useradd --system --uid 10001 vision
USER vision

EXPOSE 8000
# 접근 로그는 __main__ 에서 끈다(경로에 visionKey 가 있다). 코어 덤프는 compose/RunPod 에서 ulimit 0.
CMD ["python", "-m", "pair_vision"]
