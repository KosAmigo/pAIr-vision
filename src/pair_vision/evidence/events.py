"""RuleEvent -> 콜백 본문 (계약 §3-4)."""

from __future__ import annotations

import hashlib

from pair_vision.rules.hands import RuleEvent

# TABLET·NOTE·SMARTWATCH 는 휴대폰 채널(PHONE_CAM)로만 만든다(2026-11-26 확정).
SOURCE_BY_TYPE = {
    "HANDS_OUT": "PHONE_CAM",
    "HAND_OUT_ONE": "PHONE_CAM",
    "TABLET": "PHONE_CAM",
    "NOTE": "PHONE_CAM",
    "SMARTWATCH": "PHONE_CAM",
    # 웹캠(pcRules 포팅). TABLET·NOTE·SMARTWATCH 는 웹캠에서 만들지 않는다(확정 2026-11-26).
    "NO_FACE": "PC_CAM",
    "MULTI_FACE": "PC_CAM",
    "HEAD_TURN": "PC_CAM",
    "HEAD_DOWN": "PC_CAM",
    "GAZE_AWAY": "PC_CAM",
    "TORSO_SHIFT": "PC_CAM",
    "TORSO_TURN": "PC_CAM",
}


def make_client_id(session_key: str, event_type: str, started_at_ms: int) -> str:
    """"v" + hex(SHA-256(sessionKey|type|floor(startedAtMs/1000))) 앞 32자 (총 33자)."""
    raw = f"{session_key}|{event_type}|{started_at_ms // 1000}".encode()
    return "v" + hashlib.sha256(raw).hexdigest()[:32]


def build_callback_event(session_key: str, event: RuleEvent, rules_version: str) -> dict:
    body = {
        "clientId": make_client_id(session_key, event.type, event.started_at_ms),
        "sessionKey": session_key,
        "type": event.type,
        "source": SOURCE_BY_TYPE[event.type],
        "startedAtMs": event.started_at_ms,
        "durationMs": event.duration_ms,
        "detail": event.detail,
        "final": event.final,
        "rulesVersion": rules_version,
    }
    if event.confidence is not None:
        body["confidence"] = event.confidence
    return body
