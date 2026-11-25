from pair_vision.rules.config import HandRules
from pair_vision.rules.hands import HandFrame, HandRuleEngine, HandState

NORMAL = (2, 0)
BOTH_OUT = (0, 2)
ONE_OUT = (1, 1)


def feed(engine, t, state):
    return engine.feed(HandFrame(t, *state))


def run(engine, start, end, state, step=1000):
    events = []
    for t in range(start, end + 1, step):
        events += feed(engine, t, state)
    return events


def test_hands_out_fires_at_seven_seconds():
    e = HandRuleEngine(HandRules())
    assert run(e, 0, 6000, BOTH_OUT) == []
    events = feed(e, 7000, BOTH_OUT)
    assert len(events) == 1
    ev = events[0]
    assert (ev.type, ev.started_at_ms, ev.duration_ms, ev.final) == ("HANDS_OUT", 0, 7000, False)
    assert ev.detail == {"state": "BOTH_OUT", "handsSeen": 2}


def test_one_hand_out_needs_ten_seconds():
    e = HandRuleEngine(HandRules())
    assert run(e, 0, 9000, ONE_OUT) == []
    (ev,) = feed(e, 10000, ONE_OUT)
    assert ev.type == "HAND_OUT_ONE" and ev.duration_ms == 10000


def test_unseen_hands_count_as_out_by_default():
    e = HandRuleEngine(HandRules())
    assert e.classify(HandFrame(0, 0, 0)) is HandState.BOTH_OUT
    assert e.classify(HandFrame(0, 1, 0)) is HandState.ONE_OUT


def test_unseen_hands_not_out_when_configured():
    e = HandRuleEngine(HandRules(unseen_hand_counts_as_out=False))
    assert e.classify(HandFrame(0, 0, 0)) is HandState.NORMAL
    assert e.classify(HandFrame(0, 1, 0)) is HandState.NORMAL
    assert e.classify(HandFrame(0, 1, 1)) is HandState.ONE_OUT


def test_gap_within_three_seconds_continues_same_stretch():
    e = HandRuleEngine(HandRules())
    run(e, 0, 3000, BOTH_OUT)
    run(e, 4000, 5000, NORMAL)  # 2초 끊김 (<=3초)
    events = run(e, 6000, 7000, BOTH_OUT)
    assert [(x.type, x.started_at_ms, x.duration_ms) for x in events] == [("HANDS_OUT", 0, 7000)]


def test_gap_over_three_seconds_resets():
    e = HandRuleEngine(HandRules())
    run(e, 0, 3000, BOTH_OUT)
    run(e, 4000, 7000, NORMAL)  # 마지막 밖 3000, 7000-3000=4000 > 3000 -> 리셋
    assert run(e, 8000, 13000, BOTH_OUT) == []  # 새로 5초뿐
    (ev,) = feed(e, 15000, BOTH_OUT)
    assert ev.started_at_ms == 8000


def test_transition_between_abnormal_states_does_not_sum():
    e = HandRuleEngine(HandRules())
    run(e, 0, 5000, BOTH_OUT)
    assert run(e, 6000, 15000, ONE_OUT) == []
    (ev,) = feed(e, 16000, ONE_OUT)
    assert ev.type == "HAND_OUT_ONE" and ev.started_at_ms == 6000


def test_final_event_on_recovery_has_last_duration():
    e = HandRuleEngine(HandRules())
    run(e, 0, 9000, BOTH_OUT)
    events = run(e, 10000, 14000, NORMAL)
    assert len(events) == 1
    ev = events[0]
    assert ev.final and ev.duration_ms == 9000 and ev.started_at_ms == 0


def test_cooldown_suppresses_when_threshold_reached_early():
    e = HandRuleEngine(HandRules(hands_out_ms=2000, cooldown_ms=10000))
    run(e, 0, 3000, BOTH_OUT)
    run(e, 4000, 8000, NORMAL)  # final, 쿨다운은 마지막 밖(3000)+10000=13000 까지
    assert run(e, 9000, 12000, BOTH_OUT) == []  # 임계(2초)를 넘어도 쿨다운 안
    (ev,) = feed(e, 13000, BOTH_OUT)
    assert ev.type == "HANDS_OUT"


def test_flush_closes_open_event():
    e = HandRuleEngine(HandRules())
    run(e, 0, 8000, BOTH_OUT)
    (ev,) = e.flush()
    assert ev.final and ev.duration_ms == 8000
    assert e.flush() == []


def test_no_event_flushed_if_never_emitted():
    e = HandRuleEngine(HandRules())
    run(e, 0, 3000, BOTH_OUT)
    assert e.flush() == []


def test_frame_gap_longer_than_tolerance_breaks_stretch():
    e = HandRuleEngine(HandRules())
    feed(e, 0, BOTH_OUT)
    assert feed(e, 10000, BOTH_OUT) == []


def test_tick_closes_after_frames_stop():
    e = HandRuleEngine(HandRules())
    run(e, 0, 8000, BOTH_OUT)
    assert e.tick(9000) == []
    (ev,) = e.tick(12000)
    assert ev.final


def test_out_of_order_frame_ignored():
    e = HandRuleEngine(HandRules())
    feed(e, 5000, BOTH_OUT)
    assert feed(e, 4000, BOTH_OUT) == []


def test_reset_discards_state():
    e = HandRuleEngine(HandRules())
    run(e, 0, 8000, BOTH_OUT)
    e.reset()
    assert e.flush() == []
