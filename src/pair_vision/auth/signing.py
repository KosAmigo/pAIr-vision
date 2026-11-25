"""콜백 서명 (계약 §3-4): hex(HMAC-SHA256(비밀, timestamp + "." + 본문 바이트))."""

from __future__ import annotations

import hashlib
import hmac


def sign(secret: str, timestamp: str, body: bytes) -> str:
    message = timestamp.encode() + b"." + body
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def verify(secret: str, timestamp: str, body: bytes, signature: str) -> bool:
    return hmac.compare_digest(sign(secret, timestamp, body), signature)
