"""configs/models.lock 읽기·검증·생성. 가중치 파일은 저장소에 넣지 않고 sha256 만 기록한다."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml


class ModelLockError(RuntimeError):
    pass


@dataclass(frozen=True)
class LockedModel:
    name: str
    role: str
    repo: str  # 출처(HF repo id 또는 URL)
    license: str
    directory: str  # 모델 폴더 아래 위치(파일이면 파일 이름)
    files: Mapping[str, str] = field(default_factory=dict)  # 디렉터리 기준 상대 경로 -> sha256
    zip_url: str = ""  # 단일 파일 모델을 zip 으로 배포하는 곳(없으면 repo 주소가 곧 파일 주소)
    zip_member: str = ""  # zip 안의 파일 경로


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_lock(path: str | Path) -> list[LockedModel]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    models = []
    for raw in data.get("models") or []:
        models.append(
            LockedModel(
                name=raw["name"], role=raw.get("role", ""), repo=raw["repo"],
                license=raw["license"], directory=raw["directory"], files=dict(raw.get("files") or {}),
                zip_url=raw.get("zip_url", ""), zip_member=raw.get("zip_member", ""),
            )
        )
    return models


def verify_lock(models_dir: str | Path, locked: list[LockedModel]) -> None:
    """모델 폴더의 파일이 lock 의 sha256 과 같은지 확인한다. 하나라도 다르면 ModelLockError."""
    base = Path(models_dir)
    problems: list[str] = []
    for model in locked:
        for rel, expected in model.files.items():
            target = base / model.directory / rel if rel != "." else base / model.directory
            if not target.is_file():
                problems.append(f"{model.name}: 파일 없음 {target}")
            elif sha256_of(target) != expected:
                problems.append(f"{model.name}: sha256 불일치 {target}")
    if problems:
        raise ModelLockError("모델 검증 실패 - " + "; ".join(problems))


def build_lock_entry(models_dir: Path, meta: Mapping[str, str]) -> dict:
    """모델 폴더(또는 단일 파일)를 읽어 lock 항목 하나를 만든다."""
    target = models_dir / meta["directory"]
    if target.is_file():
        files = {".": sha256_of(target)}
    else:
        files = {p.relative_to(target).as_posix(): sha256_of(p) for p in sorted(target.rglob("*")) if p.is_file()}
    return {**meta, "files": files}
