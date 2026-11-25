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


def test_model_without_weights_is_skipped_not_crash(tmp_path, capsys):
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
