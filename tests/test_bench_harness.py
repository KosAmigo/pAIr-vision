from pathlib import Path

import pytest

from pair_vision.models.base import Box, Frame
from tools.bench import compare
from tools.bench.adapters import REGISTRY, FakeBench
from tools.bench.harness import (
    ObjectDet,
    RawResult,
    TimingStats,
    list_images,
    measure,
    percentile,
    summarize,
    to_detection,
)


def make_photos(folder: Path, n: int = 3) -> list[Path]:
    paths = []
    for i in range(n):
        p = folder / f"img{i}.jpg"
        p.write_bytes(b"jpeg-bytes")
        paths.append(p)
    return paths


def test_list_images_sorted_and_filtered(tmp_path):
    make_photos(tmp_path, 2)
    (tmp_path / "note.txt").write_text("x")
    assert [p.name for p in list_images(tmp_path)] == ["img0.jpg", "img1.jpg"]


def test_list_images_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_images(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        list_images(tmp_path)


def test_percentile_and_stats():
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile([], 0.5) == 0.0
    s = TimingStats.from_samples([10.0, 20.0, 30.0])
    assert (s.count, s.mean_ms, s.p50_ms, s.max_ms) == (3, 20.0, 20.0, 30.0)


def test_to_detection_picks_highest_score_keyboard():
    raw = RawResult(
        objects=(
            ObjectDet("keyboard", 0.4, Box(0, 0, 0.1, 0.1)),
            ObjectDet("keyboard", 0.9, Box(0.2, 0.5, 0.6, 0.9)),
            ObjectDet("cell phone", 0.8, Box(0, 0, 1, 1)),
        )
    )
    det = to_detection(Frame("k", "phone", 1, 0), raw)
    assert det.keyboard == Box(0.2, 0.5, 0.6, 0.9)
    assert det.mouse is None


def test_fake_bench_satisfies_detector_contract():
    dets = FakeBench().run([Frame("k", "phone", 1, 5, b"x"), Frame("k", "phone", 2, 6, b"y")])
    assert [d.seq for d in dets] == [1, 2]
    assert dets[0].keyboard is not None and dets[0].mouse is not None


def test_measure_uses_injected_clock_and_counts(tmp_path):
    photos = make_photos(tmp_path, 3)
    ticks = iter(i * 0.01 for i in range(10_000))  # 호출마다 10ms 전진
    report = measure(FakeBench(), photos, warmup=1, repeat=2, clock=lambda: next(ticks))
    assert report.images == 3
    assert report.timing.count == 6
    assert report.timing.mean_ms == pytest.approx(10.0)
    assert report.keyboard_found == 3 and report.mouse_found == 3
    assert report.label_summary["keyboard"] == (3, pytest.approx(0.9))


def test_summarize_counts_images_not_boxes():
    raws = {
        "a": RawResult(
            objects=(ObjectDet("mouse", 0.5, Box(0, 0, 1, 1)), ObjectDet("mouse", 0.7, Box(0, 0, 1, 1)))
        )
    }
    assert summarize(raws) == {"mouse": (1, pytest.approx(0.6))}


def test_model_without_weights_is_skipped_not_crash(tmp_path, capsys, monkeypatch):
    # 이 PC 에 모델을 받아 둔 경우에도 "가중치 없음" 상황을 재현하도록 모델 폴더를 빈 임시 폴더로 바꾼다
    monkeypatch.setattr("tools.bench.adapters.MODELS_DIR", tmp_path / "no-models")
    make_photos(tmp_path, 1)
    code = compare.main(["--images", str(tmp_path), "--models", "fake,rtdetr-r18"])
    out = capsys.readouterr()
    assert code == 0
    assert "| fake |" in out.out
    assert "[건너뜀] rtdetr-r18" in out.err


def test_cli_rejects_unknown_model(tmp_path):
    make_photos(tmp_path, 1)
    assert compare.main(["--images", str(tmp_path), "--models", "nope"]) == 2


def test_cli_list_and_json(tmp_path, capsys):
    assert compare.main(["--list"]) == 0
    assert "fake" in capsys.readouterr().out and "owlv2" in REGISTRY
    make_photos(tmp_path, 1)
    out = tmp_path / "r.json"
    assert compare.main(["--images", str(tmp_path), "--models", "fake", "--json", str(out)]) == 0
    assert '"fake"' in out.read_text(encoding="utf-8")


def test_label_for_text_maps_prompt_and_fragments():
    from tools.bench.adapters import DEFAULT_PROMPTS, label_for_text

    assert label_for_text("a keyboard", DEFAULT_PROMPTS) == "keyboard"
    assert label_for_text("mouse", DEFAULT_PROMPTS) == "mouse"
    assert label_for_text("smartwatch on a wrist", DEFAULT_PROMPTS) == "smartwatch"
    assert label_for_text("banana", DEFAULT_PROMPTS) == "banana"


def test_box_in_image_divides_by_photo_size_without_extra_scale():
    from tools.bench.adapters import box_in_image

    # 481x313 사진: 사진 전체를 덮는 박스는 (0,0,1,1) 이어야 한다. 예전 코드는 세로를 481/313 배로 늘려 1.54 가 나왔다
    b = box_in_image([0, 0, 481, 313], 481, 313)
    assert (b.x0, b.y0, b.x1, b.y1) == (0.0, 0.0, 1.0, 1.0)
    # 사진 절반 높이의 박스는 세로가 0.5
    b = box_in_image([100, 0, 200, 156.5], 481, 313)
    assert b.x0 == pytest.approx(100 / 481) and b.y1 == pytest.approx(0.5)


def test_hand_conf_parse_and_model_names():
    from tools.bench.adapters import MediaPipeHands
    from tools.bench.compare import build_models, parse_hand_confs

    assert parse_hand_confs("0.5,0.1") == [0.5, 0.1]
    for bad in ("abc", "0", "1.5", "", "0.5,-1"):
        with pytest.raises(ValueError):
            parse_hand_confs(bad)
    # 기준값마다 모델이 하나씩 생기고 표에서 구분되도록 이름에 값이 붙는다(기본값만 이름 그대로)
    names = [m.name for m in build_models("mediapipe-hands", [0.5, 0.1])]
    assert names == ["mediapipe-hands", "mediapipe-hands@0.1"]
    assert [m.name for m in build_models("fake", [0.5, 0.1])] == ["fake"]
    with pytest.raises(ValueError):
        MediaPipeHands(conf=0)


def test_cli_rejects_bad_hand_conf(tmp_path):
    make_photos(tmp_path, 1)
    assert compare.main(["--images", str(tmp_path), "--models", "fake", "--hand-conf", "abc"]) == 2
    assert compare.main(["--images", str(tmp_path), "--models", "fake", "--hand-conf", "0.1"]) == 0


def test_hands_are_saved_per_image(tmp_path):
    from pair_vision.models.base import HandDet

    class TwoHands(FakeBench):
        def infer(self, jpeg: bytes) -> RawResult:
            return RawResult(hands=(HandDet("left", 0.123456, 0.5), HandDet("right", 0.7, 0.4)))

    photos = make_photos(tmp_path, 2)
    report = measure(TwoHands(), photos, warmup=0)
    assert set(report.hands_per_image) == {p.name for p in photos}
    first = report.hands_per_image[photos[0].name]
    assert first == [{"side": "left", "x": 0.123, "y": 0.5}, {"side": "right", "x": 0.7, "y": 0.4}]
    assert report.hands_mean == 2.0


def test_box_in_image_clamps_padding_overflow():
    from tools.bench.adapters import box_in_image

    # OWLv2 의 패딩 영역(사진 아래·오른쪽 바깥)으로 나간 부분은 0~1 로 자른다
    b = box_in_image([-5, 300, 500, 481], 481, 313)
    assert (b.x0, b.y0, b.x1, b.y1) == (0.0, pytest.approx(300 / 313), 1.0, 1.0)
