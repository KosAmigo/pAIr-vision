"""물체 규칙(TABLET·NOTE·SMARTWATCH·그림자 MANY_HANDS)과 규칙 설정 검증."""

import pytest

from pair_vision.evidence.events import SOURCE_BY_TYPE, build_callback_event
from pair_vision.models.base import Box, Detection, HandDet, ObjectDet
from pair_vision.rules.config import DEFAULT_OBJECT_RULES, ObjectRule, parse_rules
from pair_vision.rules.objects import ObjectRuleEngine
from pair_vision.session.session import Session


def engine(**over):
    rules = dict(DEFAULT_OBJECT_RULES)
    rules.update(over)
    return ObjectRuleEngine(rules)


def test_tablet_event_after_min_duration_sampled_every_5s():
    e = engine()
    out = []
    for t in range(0, 12000, 5000):  # 0, 5000, 10000: OWLv2 주기 샘플
        out += e.feed(t, {"tablet": 0.7})
    assert [(v.type, v.final, v.duration_ms) for v in out] == [("TABLET", False, 10000)]
    assert out[0].confidence == 0.7


def test_gap_longer_than_tolerance_restarts_count():
    e = engine()
    assert e.feed(0, {"tablet": 0.6}) == []
    assert e.feed(5000, {}) == []
    assert e.feed(20000, {"tablet": 0.6}) == []  # 20초 만에 다시 보임 -> 새로 센다
    assert e.feed(25000, {"tablet": 0.6}) == []
    out = e.feed(30000, {"tablet": 0.6})
    assert len(out) == 1 and out[0].started_at_ms == 20000


def test_final_event_and_cooldown():
    e = engine(note=ObjectRule(min_ms=3000, gap_ms=2000, cooldown_ms=10000))
    out = []
    for t in range(0, 5000, 1000):
        out += e.feed(t, {"note": 0.5})
    out += e.feed(7000, {})  # 마지막으로 본 4000 + 2000 < 7000 -> 끊김
    assert [(v.final, v.duration_ms) for v in out] == [(False, 3000), (True, 4000)]
    # 쿨다운(끝난 4000 + 10000 = 14000) 안에서는 새 이벤트 없음
    again = []
    for t in range(8000, 13000, 1000):
        again += e.feed(t, {"note": 0.5})
    assert again == []


def test_flush_closes_running_event():
    e = engine(note=ObjectRule(min_ms=1000, gap_ms=2000, cooldown_ms=0))
    e.feed(0, {"note": 0.5})
    e.feed(1000, {"note": 0.9})
    out = e.flush()
    assert len(out) == 1 and out[0].final and out[0].detail["maxScore"] == 0.9


def test_out_of_order_frame_ignored():
    e = engine(note=ObjectRule(min_ms=1000, gap_ms=2000, cooldown_ms=0))
    e.feed(5000, {"note": 0.5})
    assert e.feed(4000, {"note": 0.5}) == []


def test_object_events_use_phone_cam_source():
    for t in ("TABLET", "NOTE", "SMARTWATCH"):
        assert SOURCE_BY_TYPE[t] == "PHONE_CAM"
    assert "MANY_HANDS" not in SOURCE_BY_TYPE  # 계약 밖 유형은 콜백으로 나가지 않는다


def test_callback_body_for_tablet(rules):
    e = engine(tablet=ObjectRule(min_ms=0, gap_ms=10000, cooldown_ms=0))
    ev = e.feed(1_764_000_000_000, {"tablet": 0.8})[0]
    body = build_callback_event("K" * 22, ev, rules.version)
    assert body["type"] == "TABLET" and body["source"] == "PHONE_CAM"
    assert body["rulesVersion"] == rules.version and body["confidence"] == 0.8


def phone_det(t, hands=2, objects=()):
    hs = tuple(HandDet("left", 0.3, 0.7) for _ in range(min(hands, 2)))
    return Detection(
        "K", "phone", 1, t, keyboard=Box(0.2, 0.5, 0.6, 0.9), mouse=Box(0.65, 0.6, 0.8, 0.8),
        hands=hs, objects=tuple(objects), hand_count=hands,
    )


def test_session_emits_tablet_in_exam_but_not_in_check(rules):
    profile = rules.profile("default")
    tablet = ObjectDet("tablet", 0.8, Box(0, 0, 0.2, 0.2))
    exam = Session("K", "exam", 0, profile)
    events = []
    for t in (0, 5000, 10000):
        events += exam.on_detection(phone_det(t, objects=[tablet])).events
    assert [e.type for e in events] == ["TABLET"]
    check = Session("K", "check", 0, profile)
    assert all(check.on_detection(phone_det(t, objects=[tablet])).events == [] for t in (0, 5000, 10000))


def test_three_hands_make_shadow_event_only_in_engine(rules):
    s = Session("K", "exam", 0, rules.profile("default"))
    events = []
    for t in range(0, 7000, 1000):
        events += s.on_detection(phone_det(t, hands=3)).events
    assert [e.type for e in events] == ["MANY_HANDS"]


def test_config_defaults_and_validation(rules):
    p = rules.profile("default")
    assert p.objects["tablet"].min_ms == 10000 and rules.models.owl_interval_ms == 5000
    assert rules.models.hand_threshold == 0.5 and rules.models.max_hands == 4
    base = {"version": "v", "profiles": {"default": {}}}
    with pytest.raises(ValueError, match="gap_ms"):
        parse_rules({**base, "models": {"owl_interval_ms": 10000}})  # 기본 gap 10000 < 2*10000
    with pytest.raises(ValueError, match="알 수 없는"):
        parse_rules({"version": "v", "profiles": {"default": {"objects": {"cat": {}}}}})
    with pytest.raises(ValueError, match="tablet_threshold"):
        parse_rules({**base, "models": {"tablet_threshold": 1.5}})
