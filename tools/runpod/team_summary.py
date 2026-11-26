"""총괄 시험 결과에서 `팀원4_결과요약.md`(한 장, 표 위주, 숫자와 사실만)를 만든다.

결과 폴더(ctx.results)의 파일을 읽어 채운다. 값이 없는 단계는 "실행 안 함/실패: 이유"로 적는다. 해석·권고는 쓰지 않는다.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

NAME = "팀원4_결과요약.md"
STAGES = ("decode", "dfine", "owl", "hands", "face", "pose")


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8-sig")
    except OSError:
        return None


def _rows(path: Path) -> list[dict]:
    text = _read(path)
    return list(csv.DictReader(text.splitlines())) if text else []


def _num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def last_results(results: list[tuple]) -> list[tuple]:
    """셀을 다시 돌려 같은 단계가 여러 번 쌓여도 단계 이름별로 마지막 결과만 남긴다(처음 나온 순서 유지)."""
    last: dict[str, tuple] = {}
    for r in results:
        last[r[0]] = r
    return list(last.values())


def _reason(results: list[tuple], prefix: str) -> str:
    for name, status, _, detail in results:
        if name.startswith(prefix):
            return f"{status}: {detail[:160]}" if status != "성공" else "성공"
    return "실행 안 함"


def env_section(root: Path, results_dir: Path) -> list[str]:
    env = _read(results_dir / "env.txt") or ""

    def pick(prefix: str) -> str:
        m = re.search(rf"^{prefix}:\s*(.+)$", env, re.M)
        return m.group(1).strip() if m else "-"

    gpu = re.search(r"CUDA 사용 가능: (\S+) (.*?) \(torch", env)
    commit = (_read(root / "COMMIT") or "-").strip()[:10]
    if not env:
        gpu_text = "-"
    elif gpu and gpu.group(1) == "True":
        gpu_text = gpu.group(2).strip() or "GPU"
    else:
        gpu_text = "없음(CPU)"
    return [
        "## ① 실행 환경", "",
        "| 항목 | 값 |", "|---|---|",
        f"| GPU | {gpu_text} |",
        f"| CPU 코어(os 가 보고한 값) | {pick('코어')} |",
        f"| 실효 CPU(cgroup·affinity 반영) | {pick('실효CPU')} |",
        f"| affinity / cgroup | {pick('affinity')} / {pick('cgroup')} |",
        f"| torch / transformers / mediapipe | {pick('torch')} / {pick('transformers')} / {pick('mediapipe')} |",
        f"| 날짜 | {pick('날짜')} |",
        f"| 커밋 | {commit} |",
        "",
    ]


def steps_section(results: list[tuple]) -> list[str]:
    lines = ["## ② 단계별 결과", "", "| 단계 | 결과 | 시간(초) | 비고 |", "|---|---|---|---|"]
    for name, status, secs, detail in results:
        note = "" if status == "성공" else detail.replace("|", "/")[:120]
        lines.append(f"| {name} | {status} | {secs:.0f} | {note} |")
    return lines + [""]


def diag_section(results_dir: Path, results: list[tuple]) -> list[str]:
    lines = ["## ③ 진단: 스레드·배치·fp16 (workers 1, 웹캠 off)", ""]
    text = _read(results_dir / "diag.json")
    if not text:
        return lines + [f"- {_reason(results, '⑥-0')}", ""]
    data = json.loads(text)
    lines += ["| 설정 | 스레드 | 배치 | fp16 | fps | GPU % | ms 합(단계) | dfine ms | owl ms | 비고 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for e in data["entries"]:
        cells = [e["name"], e["threads"] or "-", "on" if e["batch"] else "off", "on" if e["fp16"] else "off", e["fps"], e["gpu_util_pct"],
                 e["total_ms"], e["ms"].get("dfine"), e["ms"].get("owl"), e["note"]]
        lines.append("| " + " | ".join("-" if c is None or c == "" else str(c).replace("|", "/") for c in cells) + " |")
    c = data["chosen"]
    return lines + ["", f"- 고른 설정: {c['name']} (스레드 {c['threads_spec']}, 배치 {'on' if c['batch'] else 'off'}, fp16 {'on' if c['fp16'] else 'off'}) → 용량 측정이 이 설정을 씀", ""]


def capacity_section(results_dir: Path, results: list[tuple]) -> list[str]:
    lines = ["## ④ 용량 (응시자당 0.9fps 유지 최대 인원)", ""]
    files = sorted(results_dir.glob("capacity_*.csv"))
    rows = [r for f in files for r in _rows(f) if r.get("ok") != "skipped" and r.get("fps_min") != ""]
    if not rows:
        return lines + [f"- {_reason(results, '⑥ ')}", ""]
    combos: dict[tuple, dict] = {}
    for r in rows:
        key = (r["workers"], r["batch"], r["fp16"], r["webcam"], r.get("threads", ""))
        c = combos.setdefault(key, {"max": 0, "last": r})
        if r["ok"] == "yes":
            c["max"] = max(c["max"], int(r["examinees"]))
        if int(r["examinees"]) >= int(c["last"]["examinees"]):
            c["last"] = r
    lines += ["| workers | 스레드/워커 | 배치 | fp16 | 웹캠 | 최대 인원 | 마지막 측정 인원 | fps(휴대폰/웹캠) | p95 지연 ms | 버림 % | GPU % | CPU % |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, c in combos.items():
        r = c["last"]
        lines.append(
            f"| {key[0]} | {key[4] or '-'} | {key[1]} | {key[2]} | {key[3]} | {c['max'] if c['max'] else '미달'} | {r['examinees']} | "
            f"{r['fps_phone'] or '-'}/{r['fps_webcam'] or '-'} | {r['lat_p95_ms'] or '-'} | {r['drop_pct'] or '-'} | {r['gpu_util_pct'] or '-'} | {r['cpu_pct'] or '-'} |"
        )
    best_key, best = max(combos.items(), key=lambda kv: kv[1]["max"])
    lines += ["", f"- 가장 많은 인원을 유지한 조합: workers {best_key[0]}, 스레드 {best_key[4] or '-'}, 배치 {best_key[1]}, fp16 {best_key[2]}, 웹캠 {best_key[3]} → {best['max']}명" if best["max"]
              else "- 0.9fps 를 유지한 조합 없음(측정한 가장 작은 인원에서도 미달)"]
    pairs = []
    for (w, b, f, cam, t), c in combos.items():
        if cam == "server" and (w, b, f, "off", t) in combos:
            pairs.append(f"workers {w}/배치 {b}/fp16 {f}: server {c['max']}명 vs off {combos[(w, b, f, 'off', t)]['max']}명")
    lines.append("- 웹캠 server vs off: " + ("; ".join(pairs) if pairs else "같은 조합의 비교 없음(웹캠 server·off 를 모두 측정하지 않음)"))
    stage = [(float(r[f"ms_{s}"]), s, k) for k, c in combos.items() for r in [c["last"]] for s in STAGES if _num(r.get(f"ms_{s}")) is not None]
    if stage:
        ms, name, key = max(stage)
        lines.append(f"- 단계 ms 중 가장 큰 것: {name} {ms:.1f}ms/장 (workers {key[0]}, 스레드 {key[4] or '-'}, 배치 {key[1]}, fp16 {key[2]}, 웹캠 {key[3]})")
    else:
        lines.append("- 단계 ms: 기록 없음")
    return lines + [""]


def compare_section(results_dir: Path, results: list[tuple]) -> list[str]:
    lines = ["## ⑤ 모델 속도 (ms/장)", ""]
    text = _read(results_dir / "compare.json")
    if not text:
        return lines + [f"- {_reason(results, '⑤')}", ""]
    data = json.loads(text)
    lines += ["| 모델 | GPU | 평균 | p50 | p95 | 사진 수 |", "|---|---|---|---|---|---|"]
    for r in data["reports"]:
        t = r["timing"]
        lines.append(f"| {r['model']} | {'예' if r['gpu'] else '아니오'} | {t['mean_ms']:.1f} | {t['p50_ms']:.1f} | {t['p95_ms']:.1f} | {r['images']} |")
    return lines + [""]


def eval_section(results_dir: Path, results: list[tuple]) -> list[str]:
    lines = ["## ⑥ 정확도 (기준값별 precision/recall)", ""]
    parts = [("내 사진", "mine"), ("공개 extra", "extra")]
    shown = False
    for label, tag in parts:
        rows = _rows(results_dir / f"eval_{tag}.csv")
        if not rows:
            continue
        shown = True
        thresholds = sorted({r["threshold"] for r in rows}, key=float)
        by: dict[tuple, dict] = {}
        for r in rows:
            by.setdefault((r["model"], r["class"]), {})[r["threshold"]] = f"{r['precision']}/{r['recall']}"
        lines += [f"**{label}** (칸: precision/recall)", "", "| 모델 | 클래스 | " + " | ".join(f"≥{t}" for t in thresholds) + " |",
                  "|---|---|" + "---|" * len(thresholds)]
        for (model, cls), vals in by.items():
            lines.append(f"| {model} | {cls} | " + " | ".join(vals.get(t, "-") for t in thresholds) + " |")
        md = _read(results_dir / f"eval_{tag}.md") or ""
        counts = [ln for ln in md.splitlines() if re.match(r"\| .* \| (hands|persons) \|", ln)]
        if counts:
            lines += ["", "개수 항목(모델 | 항목 | 기준값 | 사진 수 | 정확히 맞음 | 평균 오차):", *counts]
        lines.append("")
    if not shown:
        lines += [f"- {_reason(results, '⑦')}", ""]
    return lines


def scan_section(results_dir: Path, results: list[tuple]) -> list[str]:
    lines = ["## ⑦ 방 스캔 API", ""]
    text = _read(results_dir / "scan.json")
    if not text:
        return lines + [f"- {_reason(results, '⑧')}", ""]
    data = json.loads(text)
    body = data["response"]
    reasons = "; ".join(f"{r['code']} {r['names']}" for r in body["verdict"]["reasons"]) or "없음"
    agree = data.get("persons_agreement") or {}
    return lines + [
        "| 항목 | 값 |", "|---|---|",
        f"| 사진 수 | {data['photos']} |",
        f"| 사진 출처 | {data.get('source', '기록 없음')}{' (합성 사진이라 검출 결과는 의미 없음)' if '합성' in data.get('source', '') else ''} |",
        *([f"| 사람 수 정답 일치 | {agree['match']}/{agree['labeled']}장 |"] if agree.get("labeled") else ["| 사람 수 정답 일치 | 정답 없음 |"]),
        f"| verdict | {body['verdict']['status']} |",
        f"| 사유 | {reasons} |",
        f"| 서버 처리 ms(첫 호출 / 둘째 호출) | {data['calls_ms'][0]} / {body['ms']} |",
        f"| summary | {json.dumps(body['summary'], ensure_ascii=False)} |",
        "",
    ]


def replay_section(results_dir: Path, results: list[tuple]) -> list[str]:
    lines = ["## ⑧ 스트림·콜백 시험", ""]
    text = _read(results_dir / "replay.md")
    if not text:
        return lines + [f"- {_reason(results, '⑨')}", ""]
    keep = [ln.strip() for ln in text.splitlines()
            if ln.startswith(("readyz", "스트림", "응답 집계", "end 로", "  pair_vision", "콜백 수신기"))]
    return lines + ["```", *keep, "```", ""]


def build(root: Path, results_dir: Path, results: list[tuple]) -> str:
    results = last_results(results)
    out = ["# 팀원 4용 결과 요약", "", "숫자와 사실만 적었습니다(해석·권고 없음).", ""]
    out += env_section(root, results_dir) + steps_section(results)
    out += diag_section(results_dir, results) + capacity_section(results_dir, results) + compare_section(results_dir, results)
    out += eval_section(results_dir, results) + scan_section(results_dir, results) + replay_section(results_dir, results)
    return "\n".join(out) + "\n"


def write(root: Path, results_dir: Path, results: list[tuple]) -> str:
    text = build(Path(root), Path(results_dir), results)
    (Path(results_dir) / NAME).write_text(text, encoding="utf-8")
    return text
