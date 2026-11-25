"""응시자·채널마다 큐 길이 1(최신만). 같은 seq 는 버린다 (계약 §3-2, §3-6)."""

from __future__ import annotations

from collections import deque

from pair_vision.session.frames import ParsedFrame

SEEN_SEQ_WINDOW = 64


class Mailboxes:
    def __init__(self) -> None:
        self._latest: dict[tuple[str, str], ParsedFrame] = {}
        self._seen: dict[tuple[str, str], deque[int]] = {}

    def put(self, frame: ParsedFrame) -> str | None:
        """넣는다. 버린 프레임이 있으면 이유를 돌려준다: 'duplicate'(새 프레임을 버림) | 'replaced'(옛 프레임을 버림)."""
        slot = (frame.key, frame.channel)
        seen = self._seen.setdefault(slot, deque(maxlen=SEEN_SEQ_WINDOW))
        if frame.seq in seen:
            return "duplicate"
        seen.append(frame.seq)
        replaced = slot in self._latest
        self._latest[slot] = frame
        return "replaced" if replaced else None

    def take_all(self) -> list[ParsedFrame]:
        frames = list(self._latest.values())
        self._latest.clear()
        return frames

    def drop_key(self, key: str) -> None:
        for slot in [s for s in self._latest if s[0] == key]:
            del self._latest[slot]
        for slot in [s for s in self._seen if s[0] == key]:
            del self._seen[slot]
