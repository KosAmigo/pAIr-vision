"""총괄 시험 도구(tools/runpod)의 순수 부분: 사진 찾기·정답 나누기·팀원용 요약."""

import json
from pathlib import Path

from tools.runpod import all_tests as T
from tools.runpod import team_summary


def png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG fake")


def test_photo_dirs_subfolders_and_extra(tmp_path):
    ctx = T.Ctx(tmp_path / "pair-all-tests")
    for rel in ("phone/a.png", "webcam/w.png", "scan/s.png", "extra/e.png", "extra/a.png"):
        png(ctx.root / "photos" / rel)
    dirs = T.photo_dirs(ctx)
    assert sorted(p.name for p in dirs["phone"].iterdir()) == ["a.png", "e.png", "extra_a.png"]  # 이름 충돌은 폴더 이름을 앞에
    assert {"webcam", "scan"} <= set(dirs)
    assert ctx.extra_names == {"e.png", "extra_a.png"}


def test_photo_dirs_flat_folder_is_all_phone_and_synthetic_fallback(tmp_path):
    flat = T.Ctx(tmp_path / "a" / "pair-all-tests")
    png(flat.root / "photos" / "1.png")
    png(flat.root / "photos" / "extra" / "x.png")
    assert sorted(p.name for p in T.photo_dirs(flat)["phone"].iterdir()) == ["1.png", "x.png"]
    none = T.Ctx(tmp_path / "b" / "pair-all-tests")
    if not T.photo_root(none):  # /workspace/photos 가 없는 PC 에서만 합성으로 떨어진다
        assert not T.has_real_photos(none)


def test_split_labels_by_source(tmp_path):
    labels = tmp_path / "labels.csv"
    labels.write_text("file,channel,keyboard\na.png,phone,1\ne.png,phone,0\n", encoding="utf-8")
    out = tmp_path / "o.csv"
    assert T._split_labels(labels, {"e.png"}, out, keep_in=True) == 1
    assert "e.png" in out.read_text(encoding="utf-8-sig") and "a.png" not in out.read_text(encoding="utf-8-sig")
    assert T._split_labels(labels, {"e.png"}, out, keep_in=False) == 1


def test_team_summary_with_and_without_results(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    steps = [("① 받기", "성공", 3.0, ""), ("⑥ 용량 측정", "실패", 1.0, "조합 실패")]
    text = team_summary.build(tmp_path, results, steps)
    assert "실행 안 함" in text or "실패: 조합 실패" in text
    assert "## ③ 용량" in text and "## ⑥ 방 스캔 API" in text

    (results / "capacity_x.csv").write_text(
        "workers,batch,fp16,webcam,examinees,fps_phone,fps_webcam,fps_min,ok,lat_p95_ms,drop_pct,gpu_util_pct,cpu_pct,ms_dfine,ms_hands\n"
        "2,on,on,server,10,1.0,1.0,1.0,yes,300,0,40,50,12.5,30.1\n"
        "2,on,on,server,20,0.5,0.5,0.5,no,900,10,60,90,13.0,35.0\n"
        "2,on,on,off,20,1.0,,1.0,yes,200,0,50,60,12.0,29.0\n", encoding="utf-8")
    (results / "scan.json").write_text(json.dumps({"photos": 3, "calls_ms": [900, 700], "response": {
        "ms": 700, "summary": {"maxPersons": 1}, "verdict": {"status": "WARN", "reasons": [{"code": "PHONE", "names": ["h01"]}]}}}))
    text = team_summary.build(tmp_path, results, steps)
    assert "→ 20명" in text  # 웹캠 off 가 20명 유지
    assert "server 10명 vs off 20명" in text
    assert "hands 35.0ms" in text
    assert "WARN" in text and "PHONE" in text
