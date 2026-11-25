"""파이프라인: 프레임 수신(ingest)·모델 실행·결과 전달·상태 메시지 주기 발송."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Callable

from pair_vision.evidence.callback import EventSink
from pair_vision.evidence.events import build_callback_event
from pair_vision.metrics import Metrics
from pair_vision.models.base import PHONE, Detector, Frame
from pair_vision.rules.hands import RuleEvent
from pair_vision.session.frames import FrameError, ParsedFrame, parse_frame
from pair_vision.session.mailbox import Mailboxes
from pair_vision.session.registry import SessionRegistry

log = logging.getLogger("pair_vision.pipeline")

UNKNOWN_SESSION_INTERVAL_S = 10.0
ERROR_THROTTLE_MAX_KEYS = 10_000
MAX_BATCH = 16

Reply = Callable[[dict], None]


def key_tag(key: str) -> str:
    """로그용 짧은 식별자. 키 자체는 로그에 남기지 않는다."""
    return hashlib.sha256(key.encode()).hexdigest()[:8]


class Hub:
    """연결된 /v1/stream 소켓들에게 텍스트 메시지를 뿌린다."""

    def __init__(self) -> None:
        self._queues: set[asyncio.Queue[dict]] = set()

    def subscribe(self) -> asyncio.Queue[dict]:
        q: asyncio.Queue[dict] = asyncio.Queue(maxsize=1000)
        self._queues.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict]) -> None:
        self._queues.discard(q)

    def publish(self, message: dict) -> None:
        for q in self._queues:
            if q.full():
                q.get_nowait()
            q.put_nowait(message)


class Pipeline:
    def __init__(
        self,
        registry: SessionRegistry,
        detector: Detector,
        hub: Hub,
        sink: EventSink,
        metrics: Metrics,
        rules_version: str,
        max_frame_bytes: int,
    ) -> None:
        self.registry, self.detector, self.hub, self.sink = registry, detector, hub, sink
        self.metrics, self.rules_version = metrics, rules_version
        self.max_frame_bytes = max_frame_bytes
        self.mailboxes = Mailboxes()
        self.processed = 0
        self._wake = asyncio.Event()
        self._unknown_last: dict[str, float] = {}
        self._bad_last: dict[str, float] = {}
        self._queued_at: dict[tuple[str, str], float] = {}

    # --- 수신 ---
    def _throttled(self, table: dict[str, float], key: str, now: float) -> bool:
        if now - table.get(key, -1e9) < UNKNOWN_SESSION_INTERVAL_S:
            return True
        if len(table) >= ERROR_THROTTLE_MAX_KEYS:
            table.clear()
        table[key] = now
        return False

    def ingest(self, data: bytes, reply: Reply) -> None:
        now = time.monotonic()
        try:
            frame: ParsedFrame = parse_frame(data, self.max_frame_bytes)
        except FrameError as err:
            self.metrics.frames_dropped.labels("bad_frame").inc()
            if err.key and not self._throttled(self._bad_last, err.key, now):
                reply({"type": "error", "code": err.code, "sessionKey": err.key})
            return
        self.metrics.frames_received.labels(frame.channel).inc()
        session = self.registry.get(frame.key)
        if session is None:
            self.metrics.frames_dropped.labels("unknown_session").inc()
            if not self._throttled(self._unknown_last, frame.key, now):
                reply({"type": "error", "code": "UNKNOWN_SESSION", "sessionKey": frame.key})
            return
        session.note_frame_arrival(frame.channel, now)
        dropped = self.mailboxes.put(frame)
        if dropped:
            self.metrics.frames_dropped.labels(dropped).inc()
        if dropped != "duplicate":
            self._queued_at[(frame.key, frame.channel)] = now
            self._wake.set()

    # --- 처리 ---
    async def run(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            await self.process_pending()

    async def process_pending(self) -> int:
        frames = self.mailboxes.take_all()
        done = 0
        for i in range(0, len(frames), MAX_BATCH):
            done += await self._process_batch(frames[i : i + MAX_BATCH])
        return done

    async def _process_batch(self, parsed: list[ParsedFrame]) -> int:
        batch: list[Frame] = []
        for p in parsed:
            session = self.registry.get(p.key)
            if session is None:
                continue
            if session.origin_ms is None and p.channel == PHONE:
                session.origin_ms = p.recv_ms
            batch.append(Frame(p.key, p.channel, p.seq, p.recv_ms, p.jpeg, session.origin_ms or p.recv_ms))
        if not batch:
            return 0
        try:
            detections = await asyncio.to_thread(self.detector.run, batch)
        except Exception:  # noqa: BLE001 - 모델 오류가 서버를 죽이지 않게
            self.metrics.detector_errors.inc()
            log.exception("모델 실행 오류")
            return 0
        now = time.monotonic()
        for det in detections:
            session = self.registry.get(det.key)
            if session is None:
                continue
            out = session.on_detection(det, now)
            self.processed += 1
            self.metrics.frames_processed.labels(det.channel).inc()
            queued = self._queued_at.pop((det.key, det.channel), None)
            if queued is not None:
                self.metrics.frame_latency.observe(max(0.0, now - queued))
            self.emit(det.key, out.events)
            if out.check is not None:
                self.hub.publish({"type": "check", "sessionKey": det.key, **out.check})
        return len(detections)

    def emit(self, session_key: str, events: list[RuleEvent]) -> None:
        for ev in events:
            self.metrics.events.labels(ev.type, str(ev.final).lower()).inc()
            self.sink.enqueue(build_callback_event(session_key, ev, self.rules_version))

    # --- 상태 메시지 / 끊긴 세션 점검 ---
    def tick_sessions(self) -> None:
        mono = time.monotonic()
        for session in self.registry.all():
            self.emit(session.key, session.tick(mono))
            if session.mode == "exam":
                self.hub.publish(session.state_message(mono))

    async def state_loop(self, interval_s: float) -> None:
        while True:
            await asyncio.sleep(interval_s)
            self.tick_sessions()
