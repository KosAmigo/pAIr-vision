"""응시자 한 명의 세션: 모드, 보정, 규칙 엔진, 상태 메시지."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from pair_vision.models.base import PHONE, WEBCAM, Box, Detection, WebcamFeatures
from pair_vision.rules.check import CheckEvaluator
from pair_vision.rules.config import Profile
from pair_vision.rules.geometry import to_hand_frame
from pair_vision.rules.hands import HandRuleEngine, RuleEvent
from pair_vision.rules.objects import ObjectRuleEngine
from pair_vision.rules.webcam import (
    WEBCAM_EVENT_TYPES,
    WebcamBaseline,
    build_baseline,
    webcam_conditions,
)

IDLE_AFTER_S = 5.0  # 계약: idle = 최근 5초 프레임 없음
FPS_WINDOW_S = 10.0


@dataclass
class SessionOutput:
    events: list[RuleEvent] = field(default_factory=list)
    check: dict | None = None


class Session:
    def __init__(self, key: str, mode: str, started_at_ms: int | None, profile: Profile) -> None:
        self.key = key
        self.mode = mode
        self.started_at_ms = started_at_ms
        self.profile = profile
        self.engine = HandRuleEngine(profile.hands)
        self.objects = ObjectRuleEngine(profile.objects)
        self.webcam_rules = ObjectRuleEngine(profile.webcam.rules, WEBCAM_EVENT_TYPES)
        self._baseline: WebcamBaseline | None = None
        self._calib_samples: list[WebcamFeatures] = []
        self.checker = CheckEvaluator(profile.check, profile.hands.area_margin)
        self.origin_ms: int | None = None  # 첫 휴대폰 프레임 recvMs (가짜 시나리오 기준점)
        self.last_recv_ms: int | None = None
        self.last_frame_mono: dict[str, float] = {}
        self._times: dict[str, deque[float]] = {WEBCAM: deque(), PHONE: deque()}
        self._calib_start_ms: int | None = None
        self.calibration = "pending"
        self.last_analyzed_ms = 0
        self._keyboard: Box | None = None  # 정지 물체 박스 캐시(설계 §7-3)
        self._mouse: Box | None = None
        self._tick_ref: tuple[int, float] | None = None

    def recalibrate(self) -> None:
        self.calibration = "pending"
        self._calib_start_ms = None
        self.engine.reset()
        self.objects.reset()
        self.webcam_rules.reset()
        self._baseline = None
        self._calib_samples = []
        self.checker.reset()

    def switch_to_exam(self, started_at_ms: int | None) -> None:
        self.mode = "exam"
        if started_at_ms is not None:
            self.started_at_ms = started_at_ms
        self.recalibrate()

    def note_frame_arrival(self, channel: str, mono: float) -> None:
        self.last_frame_mono[channel] = mono

    def on_detection(self, det: Detection, mono: float | None = None) -> SessionOutput:
        mono = time.monotonic() if mono is None else mono
        self.last_recv_ms = det.recv_ms
        self._tick_ref = (det.recv_ms, mono)
        self.last_analyzed_ms = det.recv_ms
        times = self._times[det.channel]
        times.append(mono)
        while times and mono - times[0] > FPS_WINDOW_S:
            times.popleft()
        if det.channel == WEBCAM:
            return self._on_webcam(det)
        return self._on_phone(det)

    def _on_webcam(self, det: Detection) -> SessionOutput:
        if self.mode != "exam":
            return SessionOutput()
        if self.calibration != "done":
            self._update_calibration(det)
            return SessionOutput()
        if det.webcam is None or self._baseline is None:
            return SessionOutput()  # 분석 결과가 없는 프레임(모델 없음)은 "이상 없음"으로도 "이상"으로도 세지 않는다
        cond = webcam_conditions(det.webcam, self._baseline, self.profile.webcam.thresholds)
        present = {label: 1.0 for label, active in cond.items() if active}
        return SessionOutput(events=self.webcam_rules.feed(det.recv_ms, present))

    def _update_calibration(self, det: Detection) -> None:
        """보정: calibration_ms 동안 웹캠 특징을 모아 중앙값으로 기준 자세를 만든다.
        정면 얼굴이 충분히 안 잡히면 보정을 끝내지 않고 처음부터 다시 모은다(프론트는 오류를 띄우지만 서버는 재시도)."""
        recv_ms = det.recv_ms
        if self._calib_start_ms is None:
            self._calib_start_ms = recv_ms
        if det.webcam is not None:
            self._calib_samples.append(det.webcam)
        if recv_ms - self._calib_start_ms < self.profile.calibration_ms:
            return
        if det.webcam is None:  # 특징값이 없는 프레임(웹캠 모델 미적재)이면 보정할 수 없다
            self._calib_start_ms, self._calib_samples = None, []
            return
        baseline = build_baseline(self._calib_samples, min_samples=self.profile.webcam.calibration_min_samples)
        if baseline is None:
            self._calib_start_ms, self._calib_samples = None, []
            return
        self._baseline = baseline
        self.calibration = "done"
        self._calib_samples = []

    def _on_phone(self, det: Detection) -> SessionOutput:
        if self.mode == "check":
            return SessionOutput(check=self.checker.evaluate(det))
        self._keyboard = det.keyboard or self._keyboard
        self._mouse = det.mouse or self._mouse
        area = [b for b in (self._keyboard, self._mouse) if b is not None]
        frame = to_hand_frame(det, area, self.profile.hands.area_margin)
        events = self.engine.feed(frame)
        events += self.objects.feed(det.recv_ms, self._present_objects(det))
        return SessionOutput(events=events)

    @staticmethod
    def _present_objects(det: Detection) -> dict[str, float]:
        present = {o.label: o.score for o in det.objects if o.label in ("tablet", "note", "smartwatch")}
        count = det.hand_count if det.hand_count is not None else len(det.hands)
        if count >= 3:
            present["many_hands"] = 1.0
        return present

    def tick(self, mono: float) -> list[RuleEvent]:
        """프레임이 끊긴 세션의 이상 상태를 닫는다. recvMs 시간축은 마지막 프레임 + 경과 시간으로 추정."""
        if self._tick_ref is None or self.mode != "exam":
            return []
        recv_ms, ref_mono = self._tick_ref
        now_ms = recv_ms + int((mono - ref_mono) * 1000)
        return self.engine.tick(now_ms) + self.objects.tick(now_ms) + self.webcam_rules.tick(now_ms)

    def end(self) -> list[RuleEvent]:
        if self.mode != "exam":
            return []
        return self.engine.flush() + self.objects.flush() + self.webcam_rules.flush()

    def _fps(self, channel: str, mono: float) -> float:
        times = self._times[channel]
        recent = [t for t in times if mono - t <= FPS_WINDOW_S]
        return round(len(recent) / FPS_WINDOW_S, 2)

    def _channel_state(self, channel: str, mono: float) -> str:
        last = self.last_frame_mono.get(channel)
        return "ok" if last is not None and mono - last <= IDLE_AFTER_S else "idle"

    def state_message(self, mono: float | None = None) -> dict:
        mono = time.monotonic() if mono is None else mono
        return {
            "type": "state",
            "sessionKey": self.key,
            "calibration": self.calibration,
            "webcam": self._channel_state(WEBCAM, mono),
            "phone": self._channel_state(PHONE, mono),
            "analysisFps": {WEBCAM: self._fps(WEBCAM, mono), PHONE: self._fps(PHONE, mono)},
            "lastAnalyzedMs": self.last_analyzed_ms,
        }
