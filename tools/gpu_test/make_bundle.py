"""GPU 합격 시험 산출물 만들기: 서버 코드 + 모델을 묶은 zip 과, 그 zip 의 크기·sha256 을 박은 주피터 노트북 1개.

  python tools/gpu_test/make_bundle.py --out C:/dev/pAIr/dist/vision-gpu-test
- zip 에는 서버 코드(src)·설정(configs)·설치/실행 스크립트(tools/runpod·tools/replay·tools/gpu_test)·모델(configs/models.lock 에
  있는 파일만)과 bundle-info.json(파일별 sha256)이 들어간다. 응시자 사진·키·비밀·.env·가상환경·캐시는 넣지 않는다.
- 모델 파일 sha256 을 lock 과 대조한 뒤에만 묶는다(다르면 중단).
- 노트북은 tools/gpu_test/notebook_cells.py 의 칸들로 만든다(zip 과 짝: 크기·sha256 이 노트북에 박힌다).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pair_vision.models.lock import load_lock, verify_lock  # noqa: E402

BUNDLE_NAME = "pair-vision-gpu-test.zip"
NOTEBOOK_NAME = "pair_vision_gpu_test.ipynb"
CODE_GLOBS = (
    "pyproject.toml", "uv.lock", "README.md",
    "configs/models.lock", "configs/rules.default.yaml", "configs/rules.local-cpu.yaml",
    "src/pair_vision/**/*.py",
    "tools/runpod/*.sh", "tools/runpod/*.py",
    "tools/replay/sink.py", "tools/replay/replay.py",
    "tools/gpu_test/load_test.py", "tools/gpu_test/preflight.py", "tools/gpu_test/make_images.py",
    "tools/capacity_sim/capacity_sim.py",
)
FORBIDDEN = (".env", "secret", "credential", ".pem", ".key", "photos", "bench-photos")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_files() -> list[Path]:
    files: set[Path] = set()
    for pattern in CODE_GLOBS:
        files.update(p for p in ROOT.glob(pattern) if p.is_file() and "__pycache__" not in p.parts)
    return sorted(files)


def model_files(models: Path) -> list[Path]:
    locked = load_lock(ROOT / "configs" / "models.lock")
    verify_lock(models, locked)  # 다르면 ModelLockError 로 중단
    out = []
    for m in locked:
        for rel in m.files:
            out.append(models / m.directory if rel == "." else models / m.directory / rel)
    return out


def git_version() -> str:
    try:
        rev = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True,
                               check=True).stdout.strip()
        return rev + ("+작업중변경" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_zip(out_dir: Path, models: Path) -> dict:
    codes, weights = code_files(), model_files(models)
    for p in codes:
        rel = p.relative_to(ROOT).as_posix().lower()
        if any(word in rel for word in FORBIDDEN):
            raise SystemExit(f"묶으면 안 되는 파일 이름: {rel}")
    info = {
        "made_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "code_version": git_version(),
        "rules_version": next(line.split('"')[1] for line in (ROOT / "configs/rules.default.yaml").read_text(encoding="utf-8").splitlines()
                              if line.startswith("version:")),
        "files": {},
    }
    zpath = out_dir / BUNDLE_NAME
    tmp = zpath.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in codes:
            rel = p.relative_to(ROOT).as_posix()
            data = p.read_bytes()
            info["files"][rel] = hashlib.sha256(data).hexdigest()
            z.writestr(rel, data)
        for p in weights:
            rel = "models/" + p.relative_to(models).as_posix()
            info["files"][rel] = sha256_of(p)
            z.write(p, rel, compress_type=zipfile.ZIP_STORED)  # 가중치는 거의 안 줄어든다
        z.writestr("bundle-info.json", json.dumps(info, ensure_ascii=False, indent=1))
    tmp.replace(zpath)
    return {"path": str(zpath), "bytes": zpath.stat().st_size, "sha256": sha256_of(zpath), "code_version": info["code_version"],
            "rules_version": info["rules_version"], "code_files": len(codes), "model_files": len(weights)}


def build_notebook(out_dir: Path, bundle: dict) -> Path:
    from notebook_cells import cells

    nb = {
        "cells": [
            {"cell_type": kind, "id": f"cell-{i:02d}", "metadata": {},
             "source": src.strip("\n").splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if kind == "code" else {})}
            for i, (kind, src) in enumerate(cells(bundle))
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    path = out_dir / NOTEBOOK_NAME
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--notebook-only", action="store_true", help="zip 은 그대로 두고 노트북만 다시 만든다(bundle.json 사용)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = out / "bundle.json"
    if args.notebook_only:
        bundle = json.loads(meta.read_text(encoding="utf-8"))
    else:
        bundle = build_zip(out, Path(args.models_dir))
        meta.write_text(json.dumps(bundle, ensure_ascii=False, indent=1), encoding="utf-8")
    nb = build_notebook(out, bundle)
    print(json.dumps({**bundle, "notebook": str(nb), "notebook_bytes": nb.stat().st_size,
                      "notebook_sha256": sha256_of(nb)}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
