import os

import pytest

from conftest import new_key
from pair_vision.session.frames import (
    FrameError,
    ParsedFrame,
    encode_frame,
    is_valid_key,
    key_to_raw,
    parse_frame,
    raw_to_key,
)
from pair_vision.session.mailbox import Mailboxes

JPEG = b"\xff\xd8\xff\xe0data"


def test_key_roundtrip_and_validation():
    raw = os.urandom(16)
    key = raw_to_key(raw)
    assert len(key) == 22 and key_to_raw(key) == raw and is_valid_key(key)
    assert not is_valid_key("short") and not is_valid_key("!" * 22)


def test_encode_parse_roundtrip():
    key = new_key()
    p = parse_frame(encode_frame(key, "phone", 7, 1764000000123, JPEG), 1000)
    assert (p.key, p.channel, p.seq, p.recv_ms, p.jpeg) == (key, "phone", 7, 1764000000123, JPEG)


@pytest.mark.parametrize("mutate", ["version", "channel", "magic", "short", "big"])
def test_bad_frames(mutate):
    data = bytearray(encode_frame(new_key(), "webcam", 1, 1, JPEG))
    if mutate == "version":
        data[0] = 2
    elif mutate == "channel":
        data[1] = 9
    elif mutate == "magic":
        data[30] = 0
    elif mutate == "short":
        data = data[:10]
    with pytest.raises(FrameError) as exc:
        parse_frame(bytes(data), 36 if mutate == "big" else 1000)
    assert exc.value.code == "BAD_FRAME"


def pf(key, channel, seq):
    return ParsedFrame(key, channel, seq, seq * 1000, JPEG)


def test_mailbox_keeps_only_latest_per_key_and_channel():
    m, k = Mailboxes(), new_key()
    assert m.put(pf(k, "phone", 1)) is None
    assert m.put(pf(k, "phone", 2)) == "replaced"
    assert m.put(pf(k, "webcam", 1)) is None
    frames = m.take_all()
    assert sorted((f.channel, f.seq) for f in frames) == [("phone", 2), ("webcam", 1)]
    assert m.take_all() == []


def test_mailbox_drops_duplicate_seq():
    m, k = Mailboxes(), new_key()
    m.put(pf(k, "phone", 5))
    m.take_all()
    assert m.put(pf(k, "phone", 5)) == "duplicate"
    assert m.take_all() == []


def test_mailbox_drop_key():
    m, k = Mailboxes(), new_key()
    m.put(pf(k, "phone", 1))
    m.drop_key(k)
    assert m.take_all() == [] and m.put(pf(k, "phone", 1)) is None
