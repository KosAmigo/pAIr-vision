"""환경 변수 설정. 비밀(키·서명 비밀)은 환경 변수로만 받는다."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Settings:
    api_keys: tuple[str, ...]
    callback_url: str = ""
    callback_key_id: str = "k1"
    callback_secret: str = ""
    max_sessions: int = 40  # 계약 §3-3: 실측(M4·M5) 전 임시값
    rules_path: str = "configs/rules.default.yaml"
    detector: str = "fake"
    fake_scenario: str = "normal"
    models_dir: str = "models"
    models_lock_path: str = "configs/models.lock"
    verify_models: bool = True
    workers: int = 1  # 모델 프로세스 수. 1 이면 한 프로세스(기존 동작)
    worker_threads: int = 0  # 워커마다 쓸 CPU 코어·스레드 수. 0 이면 (사용 가능 코어 / 워커 수)
    state_interval_s: float = 2.0
    max_frame_bytes: int = 256 * 1024
    version: str = "0.1.0"

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Settings:
        keys = tuple(
            k for k in (env.get("VISION_API_KEY", ""), env.get("VISION_API_KEY_NEXT", "")) if k
        )
        if not keys:
            raise ConfigError("VISION_API_KEY 환경 변수가 필요합니다.")
        url = env.get("VISION_CALLBACK_URL", "")
        secret = env.get("VISION_CALLBACK_SECRET", "")
        if url and not secret:
            raise ConfigError("VISION_CALLBACK_URL 을 쓰려면 VISION_CALLBACK_SECRET 이 필요합니다.")
        return cls(
            api_keys=keys,
            callback_url=url,
            callback_key_id=env.get("VISION_CALLBACK_KEY_ID", "k1"),
            callback_secret=secret,
            max_sessions=int(env.get("VISION_MAX_SESSIONS", "40")),
            rules_path=env.get("VISION_RULES_PATH", "configs/rules.default.yaml"),
            detector=env.get("VISION_DETECTOR", "fake"),
            fake_scenario=env.get("VISION_FAKE_SCENARIO", "normal"),
            models_dir=env.get("VISION_MODELS_DIR", "models"),
            models_lock_path=env.get("VISION_MODELS_LOCK", "configs/models.lock"),
            verify_models=env.get("VISION_VERIFY_MODELS", "1") not in ("0", "false", "False"),
            workers=max(1, int(env.get("VISION_WORKERS", "1"))),
            worker_threads=max(0, int(env.get("VISION_WORKER_THREADS", "0"))),
            state_interval_s=float(env.get("VISION_STATE_INTERVAL_S", "2")),
            max_frame_bytes=int(env.get("VISION_MAX_FRAME_BYTES", str(256 * 1024))),
            version=env.get("VISION_VERSION", "0.1.0"),
        )
