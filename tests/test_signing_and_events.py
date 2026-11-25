import hashlib
import hmac

from pair_vision.auth.bearer import BearerVerifier
from pair_vision.auth.signing import sign, verify
from pair_vision.evidence.events import build_callback_event, make_client_id
from pair_vision.rules.hands import RuleEvent


def test_signature_matches_contract_formula():
    body = '{"a":"한글"}'.encode()
    expected = hmac.new(b"s3cret", b"1764000000." + body, hashlib.sha256).hexdigest()
    assert sign("s3cret", "1764000000", body) == expected
    assert verify("s3cret", "1764000000", body, expected)
    assert not verify("s3cret", "1764000001", body, expected)
    assert not verify("other", "1764000000", body, expected)


def test_client_id_is_deterministic_33_chars():
    cid = make_client_id("KEY", "HANDS_OUT", 1764000123456)
    expected = "v" + hashlib.sha256(b"KEY|HANDS_OUT|1764000123").hexdigest()[:32]
    assert cid == expected and len(cid) == 33
    assert make_client_id("KEY", "HANDS_OUT", 1764000123999) == cid
    assert make_client_id("KEY", "HANDS_OUT", 1764000124000) != cid


def test_callback_event_shape():
    ev = RuleEvent("HANDS_OUT", 1764000123456, 7040, False, {"state": "BOTH_OUT", "handsSeen": 0})
    body = build_callback_event("KEY", ev, "2026-11-25.1")
    assert body["source"] == "PHONE_CAM" and body["rulesVersion"] == "2026-11-25.1"
    assert body["final"] is False and "confidence" not in body
    assert body["clientId"] == make_client_id("KEY", "HANDS_OUT", 1764000123456)


def test_two_bearer_keys_valid_simultaneously():
    v = BearerVerifier(("old", "new"))
    assert v.verify("Bearer old") and v.verify("Bearer new")
    assert not v.verify("Bearer nope") and not v.verify(None) and not v.verify("old")
    assert not BearerVerifier(("old", "")).verify("Bearer ")
