"""세션 등록부: 멱등 start, 용량·drain 처리."""

from __future__ import annotations

from dataclasses import dataclass

from pair_vision.rules.config import RulesConfig
from pair_vision.session.session import Session


class OverCapacityError(Exception):
    pass


class DrainingError(Exception):
    pass


class UnknownProfileError(Exception):
    pass


@dataclass(frozen=True)
class StartResult:
    session: Session
    resumed: bool


class SessionRegistry:
    def __init__(self, rules: RulesConfig, max_sessions: int) -> None:
        self._rules = rules
        self.max_sessions = max_sessions
        self.draining = False
        self._sessions: dict[str, Session] = {}

    def __len__(self) -> int:
        return len(self._sessions)

    def get(self, key: str) -> Session | None:
        return self._sessions.get(key)

    def all(self) -> list[Session]:
        return list(self._sessions.values())

    def start(
        self, key: str, mode: str, started_at_ms: int | None, profile_name: str, recalibrate: bool
    ) -> StartResult:
        existing = self._sessions.get(key)
        if existing is not None:  # 멱등: 보정·타이머 유지
            if recalibrate:
                existing.recalibrate()
            return StartResult(existing, resumed=True)
        profile = self._rules.profile(profile_name)
        if profile is None:
            raise UnknownProfileError(profile_name)
        if self.draining:
            raise DrainingError
        if len(self._sessions) >= self.max_sessions:
            raise OverCapacityError
        session = Session(key, mode, started_at_ms, profile)
        self._sessions[key] = session
        return StartResult(session, resumed=False)

    def remove(self, key: str) -> Session | None:
        return self._sessions.pop(key, None)
