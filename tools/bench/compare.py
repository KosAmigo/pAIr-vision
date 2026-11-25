"""같은 사진 폴더를 여러 모델에 돌려 속도(ms/장)와 검출 요약을 비교한다.

  uv run python -m tools.bench.compare --images C:\\photos\\desk --models fake
  uv run python -m tools.bench.compare --images <폴더> --models rtdetr-r18,dfine-s,owlv2 --json out.json
  uv run python -m tools.bench.compare --images <폴더> --models mediapipe-hands --hand-conf 0.5,0.1 --json hands.json

사용 가능한 모델 이름: --list. 가중치·라이브러리가 없는 모델은 안내만 남기고 건너뛴다.
정확도는 내지 않는다(정답 라벨 사진이 생긴 뒤 tools/eval 에서).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

from .adapters import DEFAULT_HAND_CONF, REGISTRY, MediaPipeHands
from .harness import BenchModel, ModelReport, list_images, measure, render_markdown


def device_note() -> str:
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA 없음(CPU)"
    except ImportError:
        gpu = "torch 미설치"
    return f"{platform.platform()}, Python {platform.python_version()}, {gpu}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--images", type=Path, help="사진 폴더(.jpg/.jpeg/.png, 하위 폴더 포함)")
    p.add_argument("--models", default="fake", help="쉼표로 구분한 모델 이름")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeat", type=int, default=1, help="사진당 반복 횟수")
    p.add_argument("--limit", type=int, default=0, help="앞에서 N장만(0=전부)")
    p.add_argument("--json", type=Path, help="사진별 검출(손 위치 포함)까지 담은 JSON 저장 경로")
    p.add_argument(
        "--hand-conf",
        default=str(DEFAULT_HAND_CONF),
        help="mediapipe-hands 의 손 감지 기준값(0 초과 1 이하). 쉼표로 여러 개를 주면 값마다 한 줄씩 비교한다(예: 0.5,0.1)",
    )
    p.add_argument("--list", action="store_true", help="모델 이름 목록")
    return p.parse_args(argv)


def parse_hand_confs(text: str) -> list[float]:
    try:
        values = [float(v) for v in text.split(",") if v.strip()]
    except ValueError as exc:
        raise ValueError(f"--hand-conf 는 숫자여야 합니다: {text!r}") from exc
    if not values or any(not 0.0 < v <= 1.0 for v in values):
        raise ValueError(f"--hand-conf 는 0 초과 1 이하의 숫자여야 합니다: {text!r}")
    return values


def build_models(name: str, hand_confs: list[float]) -> list[BenchModel]:
    """모델 이름 하나를 실제 모델로 만든다. mediapipe-hands 만 감지 기준값마다 하나씩 만든다."""
    if name == "mediapipe-hands":
        return [MediaPipeHands(conf=conf) for conf in hand_confs]
    return [REGISTRY[name]()]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        print("\n".join(sorted(REGISTRY)))
        return 0
    if args.images is None:
        print("--images 사진 폴더가 필요합니다.", file=sys.stderr)
        return 2
    names = [n.strip() for n in args.models.split(",") if n.strip()]
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        print(f"알 수 없는 모델: {unknown}. --list 로 확인하세요.", file=sys.stderr)
        return 2
    try:
        hand_confs = parse_hand_confs(args.hand_conf)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    images = list_images(args.images)
    if args.limit:
        images = images[: args.limit]

    reports: list[ModelReport] = []
    for name in names:
        for model in build_models(name, hand_confs):
            try:
                reports.append(measure(model, images, warmup=args.warmup, repeat=args.repeat))
            except (FileNotFoundError, ImportError) as exc:
                print(f"[건너뜀] {model.name}: {exc}", file=sys.stderr)
    if not reports:
        return 1
    note = device_note()
    print(render_markdown(reports, note))
    if args.json:
        payload = {"env": note, "reports": [asdict(r) for r in reports]}
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
