"""총괄 시험 도구(tools/runpod)의 순수 부분: 사진 찾기·정답 나누기·팀원용 요약."""

import json
import os
import shutil
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
    assert "## ④ 용량" in text and "## ⑦ 방 스캔 API" in text

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


def test_step_extra_skips_when_already_present(tmp_path, monkeypatch):
    ctx = T.Ctx(tmp_path / "pair-all-tests")
    monkeypatch.setattr(T, "PHOTO_ROOTS", ("photos", str(tmp_path / "ws" / "photos")))
    png(tmp_path / "ws" / "photos" / "extra" / "oi_1.jpg")
    (tmp_path / "ws" / "photos" / "extra" / "labels.csv").write_text("file,channel\n", encoding="utf-8")
    assert "이미 있음" in T.step_extra(ctx, "https://drive.google.com/x")  # 링크가 있어도 받지 않는다


def test_step_extra_without_url_is_skip(tmp_path, monkeypatch):
    import pytest

    ctx = T.Ctx(tmp_path / "pair-all-tests")
    monkeypatch.setattr(T, "PHOTO_ROOTS", ("photos", str(tmp_path / "ws" / "photos")))
    with pytest.raises(T.Skip):
        T.step_extra(ctx, "")


def test_step_extra_downloads_into_persistent_folder(tmp_path, monkeypatch):
    import sys
    import types
    import zipfile

    src = tmp_path / "src.zip"
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("extra/labels.csv", "file,channel\n")
        z.writestr("extra/coco/c.jpg", b"jpg")
    fake = types.SimpleNamespace(download=lambda *a, **kw: Path(kw["output"]).write_bytes(src.read_bytes()) if kw.get("id") == "x" else None)
    monkeypatch.setitem(sys.modules, "gdown", fake)
    monkeypatch.setattr(T.subprocess, "run", lambda *a, **k: None)
    ws = tmp_path / "ws" / "photos"
    monkeypatch.setattr(T, "PHOTO_ROOTS", ("photos", str(ws)))
    ctx = T.Ctx(tmp_path / "pair-all-tests")
    msg = T.step_extra(ctx, "https://drive.google.com/file/d/x/view?usp=drive_link", src.stat().st_size, T._sha256(src))
    assert (ws / "extra" / "labels.csv").is_file() and "1장" in msg
    assert "이미 있음" in T.step_extra(ctx, "https://drive.google.com/x")  # 두 번째는 받지 않는다


def test_find_labels_merges_without_duplicates(tmp_path, monkeypatch):
    ws = tmp_path / "ws" / "photos"
    monkeypatch.setattr(T, "PHOTO_ROOTS", ("photos", str(ws)))
    (ws / "extra").mkdir(parents=True)
    (ws / "labels.csv").write_text("file,channel,phone\nmine.jpg,phone,1\noi_1.jpg,phone,0\n", encoding="utf-8")
    (ws / "extra" / "labels.csv").write_text("file,channel,phone\noi_1.jpg,phone,1\n", encoding="utf-8")
    ctx = T.Ctx(tmp_path / "pair-all-tests")
    text = T.find_labels(ctx).read_text(encoding="utf-8-sig").splitlines()
    assert text[1:] == ["mine.jpg,phone,1", "oi_1.jpg,phone,1"]  # 겹친 사진은 한 번, extra 쪽 값


def test_summary_keeps_last_result_per_step(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    steps = [("⑥ 용량 측정", "실패", 1.0, "처음 실패"), ("④ pytest 전체", "성공", 2.0, ""), ("⑥ 용량 측정", "성공", 3.0, "")]
    text = team_summary.build(tmp_path, results, steps)
    assert text.count("| ⑥ 용량 측정 |") == 1 and "처음 실패" not in text
    assert [r[0] for r in team_summary.last_results(steps)] == ["⑥ 용량 측정", "④ pytest 전체"]


def test_choose_setting_prefers_fps_then_total_ms():
    e = lambda name, fps, total: {"name": name, "fps": fps, "total_ms": total}  # noqa: E731
    pick = T.choose_setting([e("a", 0.5, 100), e("b", 1.0, 300), e("c", 1.0, 250), e("d", None, None)])
    assert pick["name"] == "c"  # fps 가 높은 둘 중 ms 합이 작은 쪽
    assert T.choose_setting([e("a", 0.99, 300), e("b", 1.0, 200)])["name"] == "b"
    import pytest

    with pytest.raises(RuntimeError):
        T.choose_setting([e("x", None, None)])


def test_capacity_args_quick_and_full(tmp_path):
    ctx = T.Ctx(tmp_path / "pair-all-tests")
    ctx.gpu = True
    png(ctx.root / "photos" / "phone" / "a.png")
    choice = {"batch": True, "fp16": False, "thread_cap": 4}
    quick = T.capacity_args(ctx, ctx.results / "cap", choice)
    assert quick[quick.index("--workers") + 1] == "1,2,4" and quick[quick.index("--webcam") + 1] == "server,off"
    assert quick[quick.index("--examinees") + 1] == "10,20,30,40" and quick[quick.index("--thread-cap") + 1] == "4"
    assert quick[quick.index("--batch") + 1] == "on" and quick[quick.index("--fp16") + 1] == "off"
    ctx.quick = False
    full = T.capacity_args(ctx, ctx.results / "cap", choice)
    assert full[full.index("--workers") + 1] == "1,2,4,8" and full[full.index("--batch") + 1] == "off,on" and full[full.index("--fp16") + 1] == "off,on"


def test_diag_runs_with_fake_detector_and_feeds_capacity(tmp_path, monkeypatch):
    ctx = T.Ctx(tmp_path / "pair-all-tests", detector="fake")
    ctx.gpu = True
    (ctx.results / "env.txt").write_text("x", encoding="utf-8")
    (ctx.root / "src" / "pair_vision" / "models").mkdir(parents=True)
    src = Path(__file__).resolve().parents[1]
    shutil.copy(src / "src" / "pair_vision" / "models" / "cpuinfo.py", ctx.root / "src" / "pair_vision" / "models" / "cpuinfo.py")
    from PIL import Image

    (ctx.root / "photos" / "phone").mkdir(parents=True)
    Image.new("RGB", (64, 48), (120, 120, 120)).save(ctx.root / "photos" / "phone" / "a.png")
    monkeypatch.setattr(T, "DIAG_SCALE_GPU", ("--examinees", "2", "--duration", "1.5", "--warmup", "0.3"))
    ctx.env = lambda **extra: {**os.environ, "PYTHONPATH": os.pathsep.join([str(src / "src"), str(src)]), "PYTHONUTF8": "1", **extra}  # type: ignore[method-assign]
    msg = T.step_diag(ctx)
    assert "고른 설정" in msg and ctx.diag is not None
    diag = json.loads((ctx.results / "diag.json").read_text(encoding="utf-8"))
    assert len(diag["entries"]) == 5 and diag["cpu"]["effective"] >= 1
    assert "고른 설정" in (ctx.results / "diag.md").read_text(encoding="utf-8")
    fresh = T.Ctx(ctx.root, detector="fake")
    assert T.load_diag(fresh)["name"] == ctx.diag["name"]  # 파일에서도 이어 읽는다


def test_scan_source_and_persons_agreement(tmp_path, monkeypatch):
    ctx = T.Ctx(tmp_path / "pair-all-tests")
    ws = tmp_path / "ws" / "photos"
    monkeypatch.setattr(T, "PHOTO_ROOTS", ("photos", str(ws)))
    for i in range(20):
        png(ws / "extra" / "coco" / f"c{i:02d}.jpg")
    source, files = T._scan_source(ctx, {"phone": tmp_path})
    assert "extra/coco" in source and len(files) == 17
    source, _ = T._scan_source(ctx, {"scan": ws / "extra" / "coco"})
    assert "scan 폴더" in source
    labels = tmp_path / "l.csv"
    labels.write_text("file,persons\nc00.jpg,1\nc01.jpg,2\nc02.jpg,\n", encoding="utf-8")
    agree = T.persons_agreement(labels, files[:3], [{"persons": 1}, {"persons": 1}, {"persons": 5}])
    assert (agree["labeled"], agree["match"]) == (2, 1)


def test_team_summary_diag_and_scan_source(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "diag.json").write_text(json.dumps({"cpu": {}, "entries": [
        {"name": "스레드 자동", "threads": "8", "batch": False, "fp16": False, "fps": 1.0, "gpu_util_pct": 30, "total_ms": 200.0,
         "ms": {"dfine": 100.0, "owl": 90.0}, "note": ""}],
        "chosen": {"name": "스레드 자동", "threads_spec": "auto", "batch": False, "fp16": False}}), encoding="utf-8")
    (results / "env.txt").write_text("코어: 96\n실효CPU: 8\naffinity: 96\ncgroup: cpu.max=800000 100000\n", encoding="utf-8")
    (results / "scan.json").write_text(json.dumps({"photos": 3, "source": "합성 사진(의미 없음)", "calls_ms": [1, 2], "persons_agreement": {"labeled": 0},
        "response": {"ms": 2, "summary": {}, "verdict": {"status": "PASS", "reasons": []}}}), encoding="utf-8")
    text = team_summary.build(tmp_path, results, [])
    assert "| 실효 CPU(cgroup·affinity 반영) | 8 |" in text and "cpu.max=800000 100000" in text
    assert "## ③ 진단" in text and "고른 설정: 스레드 자동" in text
    assert "합성 사진" in text and "정답 없음" in text
