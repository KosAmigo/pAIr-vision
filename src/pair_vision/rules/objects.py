"""물체 규칙 엔진: 태블릿·노트·스마트워치(그리고 그림자 모드의 손 3개 이상)가 연속으로 보이면 이벤트.

손 규칙과 같은 방식이다(시각은 프레임 recvMs 만, 이벤트는 성립한 순간 한 번 + 끝나면 final 로 한 번 더).
- 규칙은 라벨마다 따로 센다. "보임" 관측이 들어올 때만 길이가 늘고, 마지막으로 보인 뒤 `gap_ms` 가
  지나면 끊긴 것으로 본다(OWLv2 처럼 몇 초 주기로만 검출해도 같은 규칙으로 센다).
- 쿨다운은 이벤트가 끝난(마지막으로 보인) 시점부터.
"""

from __future__ import annotations

from collections.abc import Mapping

from pair_vision.rules.config import ObjectRule
from pair_vision.rules.hands import RuleEvent

# 규칙 라벨 -> 이벤트 유형. MANY_HANDS 는 계약(§3-4)에 없는 유형이라 그림자 모드로만 쓴다.
EVENT_TYPE_BY_LABEL = {
    "tablet": "TABLET",
    "note": "NOTE",
    "smartwatch": "SMARTWATCH",
    "many_hands": "MANY_HANDS",
}


class _Track:
    def __init__(self, label: str, rule: ObjectRule, event_type: str) -> None:
        self.label, self.rule, self.event_type = label, rule, event_type
        self.reset()

    def reset(self) -> None:
        self.start_ms: int | None = None
        self.last_seen_ms = 0
        self.emitted = False
        self.max_score = 0.0
        self.samples = 0
        self.cooldown_until = 0
        self.last_t: int | None = None

    def _event(self, duration: int, *, final: bool) -> RuleEvent:
        assert self.start_ms is not None
        return RuleEvent(
            type=self.event_type,
            started_at_ms=self.start_ms,
            duration_ms=duration,
            final=final,
            detail={"label": self.label, "samples": self.samples, "maxScore": round(self.max_score, 3)},
            confidence=round(self.max_score, 3),
        )

    def close(self) -> list[RuleEvent]:
        events: list[RuleEvent] = []
        if self.start_ms is not None and self.emitted:
            events.append(self._event(self.last_seen_ms - self.start_ms, final=True))
            self.cooldown_until = self.last_seen_ms + self.rule.cooldown_ms
        self.start_ms, self.emitted, self.max_score, self.samples = None, False, 0.0, 0
        return events

    def seen(self, t: int, score: float) -> list[RuleEvent]:
        events: list[RuleEvent] = []
        if self.start_ms is not None and t - self.last_seen_ms > self.rule.gap_ms:
            events += self.close()  # 한참 안 보였다 -> 이어 세지 않는다
        if self.start_ms is None:
            self.start_ms = t
        self.last_seen_ms = t
        self.samples += 1
        self.max_score = max(self.max_score, score)
        duration = t - self.start_ms
        if not self.emitted and duration >= self.rule.min_ms and t >= self.cooldown_until:
            self.emitted = True
            events.append(self._event(duration, final=False))
        return events

    def tick(self, now_ms: int) -> list[RuleEvent]:
        if self.start_ms is not None and now_ms - self.last_seen_ms > self.rule.gap_ms:
            return self.close()
        return []


class ObjectRuleEngine:
    def __init__(self, rules: Mapping[str, ObjectRule], types: Mapping[str, str] | None = None) -> None:
        types = types or EVENT_TYPE_BY_LABEL
        self._tracks = {label: _Track(label, rule, types[label]) for label, rule in rules.items()}

    def reset(self) -> None:
        for track in self._tracks.values():
            track.reset()

    def feed(self, recv_ms: int, present: Mapping[str, float]) -> list[RuleEvent]:
        """present: 이번 프레임에서 보인 라벨 -> 점수. 순서가 뒤집힌 프레임은 무시한다."""
        events: list[RuleEvent] = []
        for label, track in self._tracks.items():
            if track.last_t is not None and recv_ms <= track.last_t:
                continue
            track.last_t = recv_ms
            if label in present:
                events += track.seen(recv_ms, present[label])
            else:
                events += track.tick(recv_ms)
        return events

    def tick(self, now_ms: int) -> list[RuleEvent]:
        return [e for track in self._tracks.values() for e in track.tick(now_ms)]

    def flush(self) -> list[RuleEvent]:
        return [e for track in self._tracks.values() for e in track.close()]
