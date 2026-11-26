"""실행 옵션·JPEG 디코드·용량 측정 도구(순수 부분) 시험."""

import io

import pytest

from pair_vision.models.base import PHONE, WEBCAM, Frame
from pair_vision.models.fake import FakeDetector
from pair_vision.models.options import RunOptions
from tools.bench import capacity


def test_options_default_matches_old_behavior():
    o = RunOptions.from_env({})
    assert (o.batch_gpu, o.fp16, o.fast_decode, o.webcam) == (False, False, False, True)
    o = RunOptions.from_env({"VISION_BATCH_GPU": "1", "VISION_FP16": "true", "VISION_WEBCAM": "off", "VISION_FAST_DECODE": "1"})
    assert (o.batch_gpu, o.fp16, o.fast_decode, o.webcam) == (True, True, True, False)


def test_webcam_off_skips_webcam_frames_in_fake():
    frames = [Frame("a", PHONE, 1, 1000, b"x"), Frame("a", WEBCAM, 1, 1000, b"x")]
    assert len(FakeDetector(options=RunOptions()).run(frames)) == 2
    out = FakeDetector(options=RunOptions(webcam=False)).run(frames)
    assert [d.channel for d in out] == [PHONE]


def test_fast_decode_draft_shrinks_big_jpeg_only():
    PIL = pytest.importorskip("PIL.Image")
    from pair_vision.models.decode import decode_jpeg

    def jpeg(w, h):
        buf = io.BytesIO()
        PIL.new("RGB", (w, h), (120, 80, 40)).save(buf, "JPEG")
        return buf.getvalue()

    big, small = jpeg(4000, 3000), jpeg(640, 480)
    assert decode_jpeg(big).size == (4000, 3000)
    assert max(decode_jpeg(big, fast=True).size) < 4000  # 1/2 로 줄여 풀림
    assert decode_jpeg(small, fast=True).size == (640, 480)  # 작은 사진은 그대로


def test_combos_and_parsing():
    combos = capacity.build_combos([1, 2], [False, True], [False], [True, False])
    assert len(combos) == 8
    assert capacity.parse_choices("off,on,off", ("off", "on")) == [False, True]
    assert capacity.parse_ints("10, 20") == [10, 20]
    assert capacity.parse_threads("0, 4,4") == [0, 4]
    assert len(capacity.build_combos([1], [True], [True], [True], [0, 4, 8])) == 3
    with pytest.raises(ValueError):
        capacity.parse_choices("maybe", ("off", "on"))


def row(workers, n, fps):
    return capacity.Row(workers, False, False, True, n, fps_min=fps)


def test_max_examinees_and_markdown():
    rows = [row(1, 10, 1.0), row(1, 20, 0.95), row(1, 30, 0.4), row(2, 10, 0.5)]
    assert capacity.max_examinees(rows) == {(1, False, False, True, 0): 20, (2, False, False, True, 0): 0}
    md = capacity.render_markdown(rows, "테스트")
    assert md.index("최대 인원") < md.index("## 상세")
    assert "20명" in md and "10명 미만" in md


def test_small_real_run_with_fake_detector(tmp_path):
    out = tmp_path / "cap"
    rc = capacity.main(["--detector", "fake", "--workers", "1", "--batch", "off", "--fp16", "off", "--webcam", "server,off",
                        "--examinees", "3", "--duration", "2", "--warmup", "0.5", "--out", str(out)])
    assert rc == 0
    text = out.with_suffix(".csv").read_text(encoding="utf-8-sig")
    assert text.count("\n") == 3 and "fps_min" in text
    assert "최대 인원" in out.with_suffix(".md").read_text(encoding="utf-8")


def test_effective_cpus_from_fake_cgroup(tmp_path, monkeypatch):
    from pair_vision.models import cpuinfo

    monkeypatch.setattr(cpuinfo.os, "cpu_count", lambda: 96)
    monkeypatch.setattr(cpuinfo.os, "sched_getaffinity", lambda _pid: set(range(96)), raising=False)
    assert cpuinfo.effective_cpus(tmp_path) == 96  # cgroup 파일이 없으면 제한 없음
    (tmp_path / "cpu.max").write_text("800000 100000\n")  # v2: 8 CPU
    assert cpuinfo.effective_cpus(tmp_path) == 8
    assert cpuinfo.default_threads(2, tmp_path) == 4 and cpuinfo.default_threads(100, tmp_path) == 1
    (tmp_path / "cpu.max").write_text("250000 100000\n")  # 2.5 -> 올림 3
    assert cpuinfo.effective_cpus(tmp_path) == 3
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert cpuinfo.effective_cpus(tmp_path) == 96
    monkeypatch.setattr(cpuinfo.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3}, raising=False)  # affinity 가 더 작으면 그쪽
    assert cpuinfo.effective_cpus(tmp_path) == 4
    assert cpuinfo.resolve_threads(0, 2, tmp_path) == 2 and cpuinfo.resolve_threads(6, 2, tmp_path) == 6


def test_effective_cpus_cgroup_v1(tmp_path, monkeypatch):
    from pair_vision.models import cpuinfo

    monkeypatch.setattr(cpuinfo.os, "cpu_count", lambda: 64)
    monkeypatch.setattr(cpuinfo.os, "sched_getaffinity", lambda _pid: set(range(64)), raising=False)
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("-1\n")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    assert cpuinfo.effective_cpus(tmp_path) == 64  # -1 = 제한 없음
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("1200000\n")
    info = cpuinfo.cpu_info(tmp_path)
    assert info.effective == 12 and "cfs_quota" in info.cgroup_raw and "실효 CPU 12" in info.describe()


def test_capacity_records_threads_and_cap(tmp_path):
    out = tmp_path / "cap"
    rc = capacity.main(["--detector", "fake", "--workers", "1,2", "--batch", "off", "--fp16", "off", "--webcam", "off",
                        "--worker-threads", "0,3", "--thread-cap", "2", "--examinees", "2", "--duration", "1.5", "--warmup", "0.3",
                        "--out", str(out)])
    assert rc == 0
    import csv

    rows = list(csv.DictReader(out.with_suffix(".csv").read_text(encoding="utf-8-sig").splitlines()))
    assert len(rows) == 4 and {"threads", "cpu_eff", "cgroup"} <= set(rows[0])
    assert {r["workers"] for r in rows} == {"1", "2"}
    md = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "스레드/워커" in md and "실효 CPU" in md
