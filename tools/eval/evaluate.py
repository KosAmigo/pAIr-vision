"""정확도 평가: 사람이 붙인 정답 CSV 와 모델 결과(`tools/bench/compare.py --json`)를 맞춰 클래스별 precision/recall 을 낸다.

흐름
  1) 빈 템플릿 만들기:   python -m tools.eval.evaluate template --photos C:\\photos --out labels.csv
  2) labels.csv 에 정답을 채운다(docs/photo-collection-guide.md 의 "정답 라벨 붙이기")
  3) 모델을 낮은 기준값으로 돌려 결과 JSON 을 만든다(점수가 낮은 검출까지 남겨야 기준값을 훑을 수 있다):
       python -m tools.bench.compare --images C:\\photos --models dfine-s,owlv2,mediapipe-hands --hand-conf 0.5,0.3,0.1 --json out.json
  4) 평가:  python -m tools.eval.evaluate score --truth labels.csv --results out.json --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --out eval

정답 CSV 열: file, channel(phone/webcam), keyboard, mouse, phone, tablet, note, smartwatch (있으면 1, 없으면 0),
            hands(보이는 손 개수), persons(사람 수), faces(얼굴 수). 칸을 비우면 그 사진·항목은 평가에서 뺀다.
기준값은 클래스마다 같은 값을 훑는다(모델이 낸 점수 >= 기준값이면 "있음"). 손은 모델 쪽 기준값(--hand-conf)이 달라서 모델 이름별 줄로 비교한다.
얼굴 수(faces)는 비교 도구에 얼굴 모델이 없어 정답 칸만 두고 아직 평가하지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

PRESENCE = ("keyboard", "mouse", "phone", "tablet", "note", "smartwatch")
COUNTS = ("hands", "persons", "faces")
FIELDS = ("file", "channel", *PRESENCE, *COUNTS)
# 모델 출력 라벨 -> 정답 클래스
LABEL_TO_CLASS = {
    "keyboard": "keyboard", "mouse": "mouse", "phone": "phone", "cell phone": "phone", "tablet": "tablet",
    "note": "note", "book": "note", "smartwatch": "smartwatch",
}
# 모델이 낼 수 있는 클래스(낼 수 없는 클래스는 평가에서 뺀다: recall 0 으로 오해하지 않게)
SUPPORTED = {
    "dfine-s": {"keyboard", "mouse", "phone", "note"},
    "rtdetr-r18": {"keyboard", "mouse", "phone", "note"},
    "rtdetr-r50": {"keyboard", "mouse", "phone", "note"},
    "owlv2": set(PRESENCE),
    "gdino-t": set(PRESENCE),
    "fake": {"keyboard", "mouse"},
}


# --- 정답 CSV ---
def write_template(photos: Path, out: Path) -> int:
    images = sorted(p for p in photos.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for p in images:
            channel = "webcam" if "webcam" in {part.lower() for part in p.relative_to(photos).parts} else "phone"
            w.writerow([p.name, channel] + [""] * (len(FIELDS) - 2))
    return len(images)


def load_truth(path: Path) -> dict[str, dict]:
    """file(이름만) -> {'channel', 클래스: int|None}. 형식이 틀리면 ValueError."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = [c for c in FIELDS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"정답 CSV 에 열이 없습니다: {missing} (필요한 열: {', '.join(FIELDS)})")
        truth: dict[str, dict] = {}
        for lineno, row in enumerate(reader, start=2):
            name = Path(row["file"].strip()).name
            if not name:
                continue
            if name in truth:
                raise ValueError(f"{lineno}행: 같은 파일이 두 번 있습니다: {name}")
            channel = row["channel"].strip().lower()
            if channel not in ("phone", "webcam"):
                raise ValueError(f"{lineno}행: channel 은 phone 또는 webcam 이어야 합니다: {row['channel']!r}")
            entry: dict = {"channel": channel}
            for col in (*PRESENCE, *COUNTS):
                raw = (row[col] or "").strip()
                if not raw:
                    entry[col] = None
                    continue
                try:
                    value = int(raw)
                except ValueError:
                    raise ValueError(f"{lineno}행 {col}: 정수여야 합니다: {raw!r}") from None
                if value < 0 or (col in PRESENCE and value > 1):
                    raise ValueError(f"{lineno}행 {col}: {'0 또는 1' if col in PRESENCE else '0 이상'} 이어야 합니다: {value}")
                entry[col] = value
            truth[name] = entry
    return truth


# --- 지표 ---
@dataclass(frozen=True)
class PRRow:
    model: str
    cls: str
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None


def predicted(objects: Iterable[dict], cls: str, threshold: float) -> bool:
    return any(LABEL_TO_CLASS.get(o["label"]) == cls and o["score"] >= threshold for o in objects)


def presence_rows(model: str, per_image: dict[str, list[dict]], truth: dict[str, dict], thresholds: Sequence[float]) -> list[PRRow]:
    classes = SUPPORTED.get(model, set(PRESENCE))
    rows: list[PRRow] = []
    for cls in PRESENCE:
        if cls not in classes:
            continue
        for t in thresholds:
            tp = fp = fn = tn = 0
            for name, objects in per_image.items():
                gt = truth.get(name)
                if gt is None or gt["channel"] != "phone" or gt[cls] is None:
                    continue
                pred = predicted(objects, cls, t)
                if gt[cls] and pred:
                    tp += 1
                elif gt[cls]:
                    fn += 1
                elif pred:
                    fp += 1
                else:
                    tn += 1
            rows.append(PRRow(model, cls, t, tp, fp, fn, tn))
    return rows


def count_rows(model: str, per_image: dict[str, list[dict]], hands: dict[str, list], truth: dict[str, dict], thresholds: Sequence[float]) -> list[dict]:
    """개수 항목: 손(모델 출력 그대로), 사람(기준값마다 person 검출 수). 정확히 맞은 비율과 평균 절대 오차."""
    out: list[dict] = []

    def summarize(cls: str, threshold: str, pairs: list[tuple[int, int]]) -> None:
        if pairs:
            out.append({
                "model": model, "class": cls, "threshold": threshold, "n": len(pairs),
                "exact": sum(1 for g, p in pairs if g == p) / len(pairs),
                "mae": sum(abs(g - p) for g, p in pairs) / len(pairs),
            })

    if "hands" in model:
        pairs = [
            (truth[n]["hands"], len(h)) for n, h in hands.items()
            if n in truth and truth[n]["hands"] is not None and truth[n]["channel"] == "phone"
        ]
        summarize("hands", "-", pairs)
    elif any(o["label"] == "person" for objs in per_image.values() for o in objs):
        for t in thresholds:
            pairs = [
                (truth[n]["persons"], sum(1 for o in objs if o["label"] == "person" and o["score"] >= t))
                for n, objs in per_image.items() if n in truth and truth[n]["persons"] is not None
            ]
            summarize("persons", f"{t:g}", pairs)
    return out


def evaluate(results: dict, truth: dict[str, dict], thresholds: Sequence[float]) -> tuple[list[PRRow], list[dict]]:
    pr: list[PRRow] = []
    counts: list[dict] = []
    for report in results["reports"]:
        model = report["model"]
        per_image = report.get("per_image", {})
        hands = report.get("hands_per_image", {})
        if "hands" not in model:
            pr += presence_rows(model, per_image, truth, thresholds)
        counts += count_rows(model, per_image, hands, truth, thresholds)
    return pr, counts


# --- 출력 ---
def _f(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}"


def render_markdown(pr: Sequence[PRRow], counts: Sequence[dict], n_truth: int) -> str:
    lines = [
        "# 정확도 평가",
        "",
        f"정답 사진 {n_truth}장. 있음/없음 항목은 휴대폰 채널 사진만 센다. `-` 는 계산할 수 없음(검출이 0개이거나 정답 '있음'이 0장).",
        "",
        "## 클래스별 precision / recall",
        "",
        "| 모델 | 클래스 | 기준값 | TP | FP | FN | TN | precision | recall |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in pr:
        lines.append(f"| {r.model} | {r.cls} | {r.threshold:g} | {r.tp} | {r.fp} | {r.fn} | {r.tn} | {_f(r.precision)} | {_f(r.recall)} |")
    if counts:
        lines += ["", "## 개수(손·사람): 정확히 맞은 비율 / 평균 절대 오차", "", "| 모델 | 항목 | 기준값 | 사진 수 | 정확히 맞음 | 평균 오차 |", "|---|---|---|---|---|---|"]
        for c in counts:
            lines.append(f"| {c['model']} | {c['class']} | {c['threshold']} | {c['n']} | {_f(c['exact'])} | {_f(c['mae'])} |")
    return "\n".join(lines) + "\n"


def write_csv(pr: Sequence[PRRow], out: Path) -> None:
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "class", "threshold", "tp", "fp", "fn", "tn", "precision", "recall"])
        for r in pr:
            w.writerow([r.model, r.cls, r.threshold, r.tp, r.fp, r.fn, r.tn, _f(r.precision), _f(r.recall)])


def parse_thresholds(text: str) -> list[float]:
    values = [float(v) for v in text.split(",") if v.strip()]
    if not values or any(not 0.0 < v <= 1.0 for v in values):
        raise ValueError(f"기준값은 0 초과 1 이하의 숫자 목록이어야 합니다: {text!r}")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template", help="사진 폴더로 빈 정답 CSV 만들기")
    t.add_argument("--photos", type=Path, required=True)
    t.add_argument("--out", type=Path, default=Path("labels.csv"))
    s = sub.add_parser("score", help="정답 CSV 와 결과 JSON 비교")
    s.add_argument("--truth", type=Path, required=True)
    s.add_argument("--results", type=Path, nargs="+", required=True, help="tools.bench.compare --json 결과(여러 개 가능)")
    s.add_argument("--thresholds", default="0.2,0.3,0.4,0.5,0.6,0.7")
    s.add_argument("--out", type=Path, default=None, help="결과 파일 접두어(.csv .md). 없으면 화면에만 출력")
    args = p.parse_args(argv)
    try:
        if args.cmd == "template":
            n = write_template(args.photos, args.out)
            print(f"{n}장 -> {args.out} (빈 칸을 채워 주세요)")
            return 0
        truth = load_truth(args.truth)
        thresholds = parse_thresholds(args.thresholds)
        merged: dict = {"reports": []}
        for path in args.results:
            merged["reports"] += json.loads(path.read_text(encoding="utf-8"))["reports"]
    except (ValueError, OSError, KeyError) as err:
        print(f"오류: {err}", file=sys.stderr)
        return 2
    pr, counts = evaluate(merged, truth, thresholds)
    md = render_markdown(pr, counts, len(truth))
    print(md)
    if args.out:
        args.out.with_suffix(".md").write_text(md, encoding="utf-8")
        write_csv(pr, args.out.with_suffix(".csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
