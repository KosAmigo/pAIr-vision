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
    with pytest.raises(ValueError):
        capacity.parse_choices("maybe", ("off", "on"))


def row(workers, n, fps):
    return capacity.Row(workers, False, False, True, n, fps_min=fps)


def test_max_examinees_and_markdown():
    rows = [row(1, 10, 1.0), row(1, 20, 0.95), row(1, 30, 0.4), row(2, 10, 0.5)]
    assert capacity.max_examinees(rows) == {(1, False, False, True): 20, (2, False, False, True): 0}
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
