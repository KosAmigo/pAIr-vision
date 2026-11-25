"""손 규칙 엔진 (계약 §9, 설계 §7). 시각은 전부 프레임의 recvMs 만 쓴다(시계 의존 없음).

- BOTH_OUT: 두 손 모두 영역 밖(안 보이는 손은 설정에 따라 "밖") 이 `hands_out_ms` 연속 -> HANDS_OUT
- ONE_OUT : 한 손만 밖                                   이 `hand_out_one_ms` 연속 -> HAND_OUT_ONE
- 정상으로 돌아와도 `gap_tolerance_ms` 까지는 같은 상태로 이어 센다(타이머는 멈추지 않고 흐른다).
- 이상 상태끼리 바뀌면(BOTH_OUT <-> ONE_OUT) 합산하지 않고 새 상태를 0부터 센다.
- 이벤트는 규칙이 성립한 순간 한 번(final=False), 이상 상태가 끝나면 마지막 길이로 한 번 더(final=True).
- 쿨다운: 이벤트가 끝난 시점부터 `cooldown_ms` 안에는 같은 엔진이 새 이벤트를 내지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pair_vision.rules.config import HandRules


class HandState(str, Enum):
    NORMAL = "NORMAL"
    BOTH_OUT = "BOTH_OUT"
    ONE_OUT = "ONE_OUT"


EVENT_TYPE = {HandState.BOTH_OUT: "HANDS_OUT", HandState.ONE_OUT: "HAND_OUT_ONE"}


@dataclass(frozen=True)
class HandFrame:
    """한 프레임의 손 상황. 안 보이는 손 수 = 2 - inside - outside."""

    recv_ms: int
    inside: int
    outside: int

    @property
    def seen(self) -> int:
        return self.inside + self.outside


@dataclass(frozen=True)
class RuleEvent:
    type: str
    started_at_ms: int
    duration_ms: int
    final: bool
    detail: dict
    confidence: float | None = None


class HandRuleEngine:
    def __init__(self, rules: HandRules) -> None:
        self._rules = rules
        self.reset()

    def reset(self) -> None:
        """진행 중이던 이상 상태·쿨다운을 모두 버린다(보정 다시 시작 때)."""
        self._state: HandState | None = None
        self._start_ms = 0
        self._last_match_ms = 0
        self._last_frame_ms: int | None = None
        self._emitted = False
        self._last_seen = 0
        self._cooldown_until = 0

    @property
    def state(self) -> HandState | None:
        return self._state

    def classify(self, frame: HandFrame) -> HandState:
        out = frame.outside
        if self._rules.unseen_hand_counts_as_out:
            out = 2 - frame.inside
        if out >= 2:
            return HandState.BOTH_OUT
        if out == 1:
            return HandState.ONE_OUT
        return HandState.NORMAL

    def _threshold(self, state: HandState) -> int:
        return self._rules.hands_out_ms if state is HandState.BOTH_OUT else self._rules.hand_out_one_ms

    def feed(self, frame: HandFrame) -> list[RuleEvent]:
        t = frame.recv_ms
        if self._last_frame_ms is not None and t <= self._last_frame_ms:
            return []  # 순서가 뒤집힌 프레임은 무시
        self._last_frame_ms = t
        new_state = self.classify(frame)
        gap = self._rules.gap_tolerance_ms
        events: list[RuleEvent] = []

        if self._state is not None:
            broken = t - self._last_match_ms > gap
            if new_state is self._state:
                if broken:  # 프레임이 한참 비었다 -> 같은 상태라도 이어 세지 않는다
                    events += self._close()
            elif new_state is HandState.NORMAL:
                if broken:
                    events += self._close()
            else:  # BOTH_OUT <-> ONE_OUT: 합산 안 함
                events += self._close()

        if new_state is not HandState.NORMAL:
            if self._state is None:
                self._state, self._start_ms, self._emitted = new_state, t, False
            self._last_match_ms = t
            self._last_seen = frame.seen
            events += self._maybe_emit(t)
        return events

    def _maybe_emit(self, t: int) -> list[RuleEvent]:
        assert self._state is not None
        duration = t - self._start_ms
        if self._emitted or duration < self._threshold(self._state) or t < self._cooldown_until:
            return []
        self._emitted = True
        return [self._event(duration, final=False)]

    def _event(self, duration: int, *, final: bool) -> RuleEvent:
        assert self._state is not None
        return RuleEvent(
            type=EVENT_TYPE[self._state],
            started_at_ms=self._start_ms,
            duration_ms=duration,
            final=final,
            detail={"state": self._state.value, "handsSeen": self._last_seen},
        )

    def _close(self) -> list[RuleEvent]:
        events: list[RuleEvent] = []
        if self._state is not None and self._emitted:
            events.append(self._event(self._last_match_ms - self._start_ms, final=True))
            self._cooldown_until = self._last_match_ms + self._rules.cooldown_ms
        self._state = None
        self._emitted = False
        return events

    def tick(self, now_ms: int) -> list[RuleEvent]:
        """프레임이 끊겼을 때를 위한 점검. now_ms 는 recvMs 와 같은 시간축이어야 한다."""
        if self._state is not None and now_ms - self._last_match_ms > self._rules.gap_tolerance_ms:
            return self._close()
        return []

    def flush(self) -> list[RuleEvent]:
        """세션 종료: 진행 중인 이상 상태를 final 로 마무리한다."""
        return self._close()
