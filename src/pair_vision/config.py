"""환경 변수 설정. 비밀(키·서명 비밀)은 환경 변수로만 받는다."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


class ConfigError(ValueError):
    pass


FP16_MODELS = ("dfine", "owlv2")


def _fp16_skip(raw: str) -> frozenset[str]:
    names = frozenset(x.strip().lower() for x in raw.split(",") if x.strip())
    unknown = names - set(FP16_MODELS)
    if unknown:
        raise ConfigError(f"VISION_FP16_SKIP 에 모르는 모델: {sorted(unknown)} (dfine, owlv2)")
    return names


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
    # 묶음 추론(모델 1프로세스, 응시자 사이 배치 + 연속 처리). 켜면 workers 는 무시한다.
    batch: bool = False
    batch_max: int = 16  # 단계마다 한 번에 돌리는 최대 프레임 수
    batch_wait_ms: float = 15.0  # GPU 가 놀 때 첫 프레임 뒤 더 모으는 시간
    prep_threads: int = 4  # JPEG 풀기·CPU 전처리 스레드 수
    owl_max_batch: int = 4  # OWLv2(960px) 묶음 상한(GPU 메모리)
    # 전부 GPU 전처리(JPEG 풀기·자르기·레터박스). None = GPU + models.fast_preprocess 일 때 자동, True/False = 강제(시험용)
    device_preprocess: bool | None = None
    fp16_skip: frozenset[str] = frozenset()  # models.fp16 이어도 fp32 로 둘 모델(dfine, owlv2). GPU 점검에서 결과가 달라질 때
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
            batch=env.get("VISION_BATCH", "0") in ("1", "true", "True"),
            batch_max=max(1, int(env.get("VISION_BATCH_MAX", "16"))),
            batch_wait_ms=max(0.0, float(env.get("VISION_BATCH_WAIT_MS", "15"))),
            prep_threads=max(1, int(env.get("VISION_PREP_THREADS", "4"))),
            owl_max_batch=max(1, int(env.get("VISION_OWL_MAX_BATCH", "4"))),
            fp16_skip=_fp16_skip(env.get("VISION_FP16_SKIP", "")),
            device_preprocess={"1": True, "0": False}.get(env.get("VISION_DEVICE_PREPROCESS", "").strip()),
            state_interval_s=float(env.get("VISION_STATE_INTERVAL_S", "2")),
            max_frame_bytes=int(env.get("VISION_MAX_FRAME_BYTES", str(256 * 1024))),
            version=env.get("VISION_VERSION", "0.1.0"),
        )
