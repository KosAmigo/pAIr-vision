import json

import pytest

from tools.eval import evaluate as ev


def write(tmp_path, rows):
    p = tmp_path / "labels.csv"
    p.write_text("file,channel,keyboard,mouse,phone,tablet,note,smartwatch,hands,persons,faces\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return p


def test_template_header_and_channel_guess(tmp_path):
    photos = tmp_path / "photos"
    (photos / "webcam").mkdir(parents=True)
    (photos / "a.jpg").write_bytes(b"x")
    (photos / "webcam" / "b.jpg").write_bytes(b"x")
    out = tmp_path / "t.csv"
    assert ev.write_template(photos, out) == 2
    lines = out.read_text(encoding="utf-8-sig").splitlines()
    assert lines[0].split(",") == list(ev.FIELDS)
    assert lines[1].startswith("a.jpg,phone") and lines[2].startswith("b.jpg,webcam")
    assert ev.load_truth(out)["a.jpg"]["keyboard"] is None  # 빈 칸 = 평가 제외


def test_bad_truth_rejected(tmp_path):
    with pytest.raises(ValueError, match="channel"):
        ev.load_truth(write(tmp_path, ["a.jpg,tablet,1,1,0,0,0,0,2,1,0"]))
    with pytest.raises(ValueError, match="0 또는 1"):
        ev.load_truth(write(tmp_path, ["a.jpg,phone,2,1,0,0,0,0,2,1,0"]))
    with pytest.raises(ValueError, match="두 번"):
        ev.load_truth(write(tmp_path, ["a.jpg,phone,1,1,0,0,0,0,2,1,0", "a.jpg,phone,1,1,0,0,0,0,2,1,0"]))


def test_precision_recall_sweep(tmp_path):
    truth = ev.load_truth(write(tmp_path, [
        "a.jpg,phone,1,1,0,0,0,0,2,1,",
        "b.jpg,phone,1,0,0,0,0,0,1,1,",
        "c.jpg,phone,0,0,0,0,0,0,0,1,",
    ]))
    results = {"reports": [{
        "model": "dfine-s",
        "per_image": {
            "a.jpg": [{"label": "keyboard", "score": 0.9}, {"label": "mouse", "score": 0.35}, {"label": "person", "score": 0.8}],
            "b.jpg": [{"label": "keyboard", "score": 0.4}, {"label": "mouse", "score": 0.6}],
            "c.jpg": [{"label": "keyboard", "score": 0.5}],
        },
        "hands_per_image": {},
    }]}
    pr, counts = ev.evaluate(results, truth, [0.3, 0.5])
    by = {(r.cls, r.threshold): r for r in pr}
    k3, k5 = by["keyboard", 0.3], by["keyboard", 0.5]
    assert (k3.tp, k3.fp, k3.fn) == (2, 1, 0)
    assert (k5.tp, k5.fp, k5.fn) == (1, 1, 1)  # b.jpg 키보드 0.4 < 0.5
    assert k5.precision == 0.5 and k5.recall == 0.5
    m5 = by["mouse", 0.5]
    assert (m5.tp, m5.fn, m5.fp, m5.precision) == (0, 1, 1, 0.0)
    assert not any(r.cls == "tablet" for r in pr)  # dfine-s 는 태블릿을 못 낸다: 평가에서 뺀다
    persons = {c["threshold"]: c for c in counts if c["class"] == "persons"}
    assert persons["0.5"]["exact"] == pytest.approx(1 / 3)


def test_hands_count_and_cli(tmp_path):
    labels = write(tmp_path, ["a.jpg,phone,,,,,,,2,,", "b.jpg,phone,,,,,,,1,,"])
    res = tmp_path / "r.json"
    res.write_text(json.dumps({"reports": [{
        "model": "mediapipe-hands@0.3", "per_image": {},
        "hands_per_image": {"a.jpg": [{}, {}], "b.jpg": [{}, {}]},
    }]}), encoding="utf-8")
    out = tmp_path / "eval"
    assert ev.main(["score", "--truth", str(labels), "--results", str(res), "--thresholds", "0.5", "--out", str(out)]) == 0
    md = out.with_suffix(".md").read_text(encoding="utf-8")
    assert "hands" in md and "0.50" in md
    assert ev.main(["score", "--truth", str(tmp_path / "nope.csv"), "--results", str(res)]) == 2
