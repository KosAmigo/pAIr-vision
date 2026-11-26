"""총괄 시험 번들(zip)과 노트북을 만든다. 이 PC(Windows)에서 실행한다.

  python tools/runpod/make_bundle.py [--photos <폴더>] [--labels labels.csv] [--drive-url <링크>] [--out dist]

만드는 것
  dist/pair-all-tests.zip      현재 HEAD 코드(git archive) + 모델 5개(lock sha256 검증) [+ photos/ + labels.csv]
  dist/pair_all_tests.ipynb    zip 의 크기·sha256(과 드라이브 링크)을 박은 총괄 노트북

zip 안은 모두 `pair-all-tests/` 아래에 들어 있다. photos 폴더 밑에 phone/ webcam/ scan/ 이 있으면 그대로 유지한다
(없으면 photos/ 전체를 휴대폰 사진으로 본다). 커밋 안 된 변경은 담기지 않으므로 먼저 커밋하세요(경고를 낸다).
다음 순서: zip 을 구글 드라이브에 올림 -> '링크가 있는 모든 사용자' 공유 -> 링크를 노트북 맨 위 DRIVE_URL 에 넣음(또는 --drive-url).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pair_vision.models.lock import ModelLockError, load_lock, verify_lock  # noqa: E402

PREFIX = "pair-all-tests"
ZIP_NAME = "pair-all-tests.zip"
NOTEBOOK_NAME = "pair_all_tests.ipynb"
TEMPLATE = ROOT / "tools" / "runpod" / "pair_all_tests.template.ipynb"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def ensure_models(models: Path) -> None:
    """setup_models.py 로 받고(이미 맞으면 건너뜀) lock sha256 으로 검증한다."""
    rc = subprocess.run([sys.executable, str(ROOT / "tools" / "runpod" / "setup_models.py"), "--models-dir", str(models)]).returncode
    if rc != 0:
        raise SystemExit("모델 받기·검증 실패(위 메시지 확인). huggingface_hub 가 없으면 pip install huggingface_hub")
    try:
        verify_lock(models, load_lock(ROOT / "configs" / "models.lock"))
    except ModelLockError as exc:
        raise SystemExit(f"모델 sha256 불일치: {exc}") from exc


def add_tree(z: zipfile.ZipFile, base: Path, arc_root: str, suffixes: tuple[str, ...] | None = None, stored: bool = False) -> int:
    count = 0
    for p in sorted(base.rglob("*")):
        if p.is_file() and (suffixes is None or p.suffix.lower() in suffixes):
            z.write(p, f"{PREFIX}/{arc_root}/{p.relative_to(base).as_posix()}",
                    compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)
            count += 1
    return count


def build_zip(out: Path, models: Path, photos: Path | None, labels: Path | None) -> None:
    code_zip = out.parent / "_code.zip"
    git("archive", "--format=zip", "-o", str(code_zip), "HEAD")
    try:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z, zipfile.ZipFile(code_zip) as code:
            for info in code.infolist():
                if info.is_dir():
                    continue
                z.writestr(f"{PREFIX}/{info.filename}", code.read(info), compress_type=zipfile.ZIP_DEFLATED)
            z.writestr(f"{PREFIX}/COMMIT", git("rev-parse", "HEAD").decode().strip())  # 결과 요약에 커밋 해시를 적으려고
            n_models = add_tree(z, models, "models", stored=True)  # 가중치는 압축이 안 되므로 그대로 담는다
            n_photos = add_tree(z, photos, "photos", IMAGE_SUFFIXES) if photos else 0
            if labels:
                z.write(labels, f"{PREFIX}/labels.csv")
        print(f"코드 {len(code.namelist())}개 + 모델 파일 {n_models}개 + 사진 {n_photos}장" + (" + labels.csv" if labels else ""))
    finally:
        code_zip.unlink(missing_ok=True)


def fill_notebook(template: Path, out: Path, size: int, sha: str, drive_url: str) -> None:
    nb = json.loads(template.read_text(encoding="utf-8"))
    values = {"ZIP_BYTES": str(size), "ZIP_SHA256": json.dumps(sha), "DRIVE_URL": json.dumps(drive_url)}
    hit: set[str] = set()
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for i, line in enumerate(cell["source"]):
            for name, value in values.items():
                if re.match(rf"{name}\s*=", line):
                    comment = line.split("#", 1)[1] if "#" in line else ""
                    cell["source"][i] = f"{name} = {value}" + (f"  #{comment}" if comment else "\n")
                    if not cell["source"][i].endswith("\n"):
                        cell["source"][i] += "\n"
                    hit.add(name)
    missing = set(values) - hit
    if missing:
        raise SystemExit(f"노트북 템플릿에서 {sorted(missing)} 줄을 찾지 못했습니다")
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "dist")
    ap.add_argument("--models-dir", type=Path, default=ROOT / "models")
    ap.add_argument("--photos", type=Path, default=None, help="(선택) 사진 폴더(하위에 phone/ webcam/ scan/ extra/ 가능). 없으면 노트북이 RunPod 의 /workspace/photos 를 찾는다")
    ap.add_argument("--labels", type=Path, default=None, help="정답 CSV(tools/eval 형식)")
    ap.add_argument("--drive-url", default="", help="구글 드라이브 공유 링크(없으면 노트북이 실행할 때 물어본다)")
    ap.add_argument("--skip-model-download", action="store_true", help="모델이 이미 있으면 받기 단계를 건너뜀(검증은 함)")
    args = ap.parse_args()

    for path, what in ((args.photos, "사진 폴더"), (args.labels, "정답 CSV")):
        if path is not None and not path.exists():
            raise SystemExit(f"{what}가 없습니다: {path}")
    if git("status", "--porcelain", "--untracked-files=no").strip():
        print("경고: 커밋 안 된 변경이 있습니다. zip 에는 HEAD 커밋만 담기고 작업 폴더의 변경은 빠집니다.")
    print("HEAD:", git("rev-parse", "--short", "HEAD").decode().strip())

    if args.skip_model_download:
        verify_lock(args.models_dir, load_lock(ROOT / "configs" / "models.lock"))
    else:
        ensure_models(args.models_dir)

    args.out.mkdir(parents=True, exist_ok=True)
    zip_path = args.out / ZIP_NAME
    build_zip(zip_path, args.models_dir, args.photos, args.labels)
    size, sha = zip_path.stat().st_size, sha256_file(zip_path)
    nb_path = args.out / NOTEBOOK_NAME
    fill_notebook(TEMPLATE, nb_path, size, sha, args.drive_url)
    print(f"\n{zip_path}\n  크기 {size:,} 바이트 ({size / 1e6:.0f}MB)\n  sha256 {sha}\n{nb_path}")
    print("\n다음: zip 을 구글 드라이브에 올리고 '링크가 있는 모든 사용자'로 공유 -> 링크를 노트북 맨 위 DRIVE_URL 에 넣기"
          + ("" if not args.drive_url else " (이미 박았음)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
