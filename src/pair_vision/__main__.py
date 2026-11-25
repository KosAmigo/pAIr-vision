"""`python -m pair_vision` — uvicorn 실행.

접근 로그를 끈다: 경로(`/v1/sessions/{visionKey}/start`)에 키가 있어 로그에 남기면 안 된다.
"""

from __future__ import annotations

import logging
import os
import sys

import uvicorn

from pair_vision.api.app import create_app
from pair_vision.config import ConfigError, Settings


def _disable_core_dumps() -> None:
    """크래시 덤프에 사진이 남지 않게 한다(설계 §5-1)."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, ValueError, OSError):
        pass


def main() -> None:
    logging.basicConfig(level=os.environ.get("VISION_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    _disable_core_dumps()
    try:
        settings = Settings.from_env(os.environ)
    except ConfigError as exc:
        sys.exit(f"설정 오류: {exc}")
    uvicorn.run(
        create_app(settings),
        host=os.environ.get("VISION_HOST", "0.0.0.0"),  # noqa: S104 - 컨테이너 안에서 실행
        port=int(os.environ.get("VISION_PORT", "8000")),
        access_log=False,
    )


if __name__ == "__main__":
    main()
