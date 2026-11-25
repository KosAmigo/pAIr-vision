"""백엔드 -> 추론 서버 바이너리 프레임 (계약 §3-2).

offset 0 uint8 version(=1) | 1 uint8 channel(1 webcam, 2 phone) | 2 uint32 BE seq |
6 uint64 BE recvMs | 14 visionKey 16바이트 | 30 JPEG
"""

from __future__ import annotations

import base64
import re
import struct
from dataclasses import dataclass

HEADER = struct.Struct(">BBIQ16s")
HEADER_SIZE = HEADER.size  # 30
JPEG_MAGIC = b"\xff\xd8\xff"
CHANNELS = {1: "webcam", 2: "phone"}
CHANNEL_IDS = {v: k for k, v in CHANNELS.items()}
_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{22}$")


class FrameError(Exception):
    def __init__(self, code: str, key: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.key = key


def raw_to_key(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def key_to_raw(key: str) -> bytes:
    return base64.urlsafe_b64decode(key + "==")


def is_valid_key(key: str) -> bool:
    """22자 base64url 이고 16바이트로 풀렸다 다시 인코딩해도 같은 값(정규형)이어야 한다."""
    if not _KEY_RE.match(key):
        return False
    try:
        return raw_to_key(key_to_raw(key)) == key
    except ValueError:
        return False


@dataclass(frozen=True)
class ParsedFrame:
    key: str
    channel: str
    seq: int
    recv_ms: int
    jpeg: bytes


def parse_frame(data: bytes, max_bytes: int) -> ParsedFrame:
    if len(data) < HEADER_SIZE:
        raise FrameError("BAD_FRAME")
    version, channel, seq, recv_ms, raw = HEADER.unpack_from(data)
    key = raw_to_key(raw)
    jpeg = data[HEADER_SIZE:]
    if (
        version != 1
        or channel not in CHANNELS
        or not jpeg.startswith(JPEG_MAGIC)
        or len(data) > max_bytes
    ):
        raise FrameError("BAD_FRAME", key)
    return ParsedFrame(key, CHANNELS[channel], seq, recv_ms, jpeg)


def encode_frame(key: str, channel: str, seq: int, recv_ms: int, jpeg: bytes) -> bytes:
    return HEADER.pack(1, CHANNEL_IDS[channel], seq, recv_ms, key_to_raw(key)) + jpeg
