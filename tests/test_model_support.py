"""가중치 없이 도는 모델 보조 코드 시험: 후처리·lock 검증·detector 선택. 실제 모델 통합 시험은 test_real_models.py."""

import hashlib
from pathlib import Path

import pytest
import yaml

from pair_vision.config import Settings
from pair_vision.models.base import Box, HandDet, ObjectDet
from pair_vision.models.fake import FakeDetector, create_detector
from pair_vision.models.lock import (
    ModelLockError,
    build_lock_entry,
    load_lock,
    sha256_of,
    verify_lock,
)
from pair_vision.models.postprocess import best_per_label, box_in_image, merge_close_hands
from pair_vision.models.real import RealDetector


def test_merge_close_hands_keeps_higher_score():
    hands = [HandDet("left", 0.50, 0.50, 0.7), HandDet("right", 0.52, 0.51, 0.95), HandDet("left", 0.1, 0.1, 0.8)]
    merged = merge_close_hands(hands, 0.06)
    assert [(h.x, h.score) for h in merged] == [(0.52, 0.95), (0.1, 0.8)]


def test_merge_keeps_three_far_hands_sorted_by_score():
    hands = [HandDet("left", 0.1, 0.1, 0.6), HandDet("right", 0.5, 0.5, 0.9), HandDet("left", 0.9, 0.9, 0.7)]
    assert [h.score for h in merge_close_hands(hands, 0.06)] == [0.9, 0.7, 0.6]


def test_box_in_image_divides_by_photo_size_and_clips():
    assert box_in_image([0, 0, 200, 150], 100, 100) == Box(0.0, 0.0, 1.0, 1.0)
    assert box_in_image([10, 20, 50, 60], 100, 200) == Box(0.1, 0.1, 0.5, 0.3)


def test_best_per_label():
    best = best_per_label([ObjectDet("note", 0.4, Box(0, 0, 1, 1)), ObjectDet("note", 0.9, Box(0, 0, 0.5, 0.5))])
    assert best["note"].score == 0.9


def test_lock_roundtrip_and_tamper_detection(tmp_path):
    (tmp_path / "m1").mkdir()
    (tmp_path / "m1" / "w.bin").write_bytes(b"weights")
    (tmp_path / "single.task").write_bytes(b"task")
    entries = [
        build_lock_entry(tmp_path, {"name": "a", "role": "r", "repo": "x/y", "license": "Apache-2.0", "directory": "m1"}),
        build_lock_entry(tmp_path, {"name": "b", "role": "r", "repo": "u", "license": "L", "directory": "single.task"}),
    ]
    assert entries[0]["files"]["w.bin"] == hashlib.sha256(b"weights").hexdigest()
    lock = tmp_path / "models.lock"
    lock.write_text(yaml.safe_dump({"models": entries}), encoding="utf-8")
    locked = load_lock(lock)
    verify_lock(tmp_path, locked)  # 통과
    (tmp_path / "m1" / "w.bin").write_bytes(b"tampered")
    with pytest.raises(ModelLockError, match="불일치"):
        verify_lock(tmp_path, locked)
    (tmp_path / "single.task").unlink()
    with pytest.raises(ModelLockError, match="파일 없음"):
        verify_lock(tmp_path, locked)
    assert sha256_of(tmp_path / "m1" / "w.bin") == hashlib.sha256(b"tampered").hexdigest()


def test_committed_lock_lists_the_confirmed_models():
    locked = load_lock(Path(__file__).resolve().parents[1] / "configs" / "models.lock")
    assert {m.name for m in locked} == {"dfine-s", "owlv2", "mediapipe-hands", "mediapipe-face", "mediapipe-pose-lite"}
    for m in locked:
        assert m.license and m.repo and all(len(h) == 64 for h in m.files.values())


def test_create_detector_selection(rules):
    assert isinstance(create_detector("fake", "normal"), FakeDetector)
    assert isinstance(create_detector("real", "normal", model_rules=rules.models), RealDetector)
    with pytest.raises(ValueError):
        create_detector("nope", "normal")


def test_real_detector_missing_weights_fails_clearly(tmp_path, rules):
    with pytest.raises(FileNotFoundError, match="모델 파일 없음"):
        RealDetector(tmp_path, rules.models).load()


def test_settings_models_env():
    s = Settings.from_env(
        {"VISION_API_KEY": "k", "VISION_DETECTOR": "real", "VISION_MODELS_DIR": "/m", "VISION_VERIFY_MODELS": "0"}
    )
    assert (s.detector, s.models_dir, s.verify_models) == ("real", "/m", False)
    assert Settings.from_env({"VISION_API_KEY": "k"}).models_dir == "models"
