"""모델 5개를 models 폴더에 내려받고 configs/models.lock 의 sha256 으로 검증한다. 몇 번 실행해도 안전하다(이미 맞으면 건너뜀).

  python tools/runpod/setup_models.py [--models-dir models]
Hugging Face(D-FINE-s, OWLv2)와 Google 저장소(MediaPipe .task 3개)에서 받는다. 받을 파일 목록과 sha256 은 lock 이 정한다.
검증이 실패하면 종료 코드 1 (모델 파일이 바뀐 것이므로 이유를 확인한 뒤 lock 을 갱신한다).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pair_vision.models.lock import ModelLockError, load_lock, sha256_of, verify_lock  # noqa: E402


def _ok(path: Path, expected: str) -> bool:
    return path.is_file() and sha256_of(path) == expected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--lock", default=str(ROOT / "configs" / "models.lock"))
    args = ap.parse_args()
    models = Path(args.models_dir)
    models.mkdir(parents=True, exist_ok=True)
    locked = load_lock(args.lock)
    for m in locked:
        for rel, expected in m.files.items():
            target = models / m.directory if rel == "." else models / m.directory / rel
            if _ok(target, expected):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"받는 중: {m.name} {rel if rel != '.' else m.directory}", flush=True)
            if m.zip_url:  # zip 으로 배포되는 단일 파일(RTMW)
                import io
                import zipfile

                data = urllib.request.urlopen(m.zip_url, timeout=120).read()  # noqa: S310 - lock 에 적힌 고정 주소
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    target.write_bytes(archive.read(m.zip_member))
            elif m.directory.endswith(".task"):
                urllib.request.urlretrieve(m.repo, target)  # noqa: S310 - lock 에 적힌 고정 주소
            else:
                from huggingface_hub import hf_hub_download

                hf_hub_download(m.repo, rel, local_dir=models / m.directory)
    try:
        verify_lock(models, locked)
    except ModelLockError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"모델 {len(locked)}개 준비·검증 완료: {models}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
