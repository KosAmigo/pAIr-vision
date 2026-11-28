"""파이프라인: 프레임 수신(ingest)·모델 실행·결과 전달·상태 메시지 주기 발송."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Callable

from pair_vision.evidence.callback import EventSink
from pair_vision.evidence.events import SOURCE_BY_TYPE, build_callback_event
from pair_vision.metrics import Metrics
from pair_vision.models.base import PHONE, Detection, Detector, Frame
from pair_vision.rules.hands import RuleEvent
from pair_vision.session.frames import FrameError, ParsedFrame, parse_frame
from pair_vision.session.mailbox import Mailboxes
from pair_vision.session.registry import SessionRegistry

log = logging.getLogger("pair_vision.pipeline")

UNKNOWN_SESSION_INTERVAL_S = 10.0
ERROR_THROTTLE_MAX_KEYS = 10_000

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
        self._inflight: set[tuple[str, str]] = set()  # 연속 처리: 모델에서 처리 중인 칸(칸마다 1장)
        self._tasks: set[asyncio.Task] = set()

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

    def drop_key(self, key: str) -> None:
        """세션 종료: 대기 프레임과 수신 시각 기록을 지운다(처리 중 프레임은 끝나면 세션이 없어 버려진다)."""
        self.mailboxes.drop_key(key)
        for slot in [s for s in self._queued_at if s[0] == key]:
            del self._queued_at[slot]

    # --- 처리 ---
    async def run(self) -> None:
        streaming = getattr(self.detector, "streaming", False)
        while True:
            await self._wake.wait()
            self._wake.clear()
            if streaming:
                self.dispatch_ready()
            else:
                await self.process_pending()

    # --- 연속 처리(묶음 추론 엔진): 프레임마다 따로 끝나고, 끝난 칸에 바로 다음 프레임을 넣는다 ---
    def dispatch_ready(self) -> int:
        """처리 중이 아닌 칸의 대기 프레임을 모델에 넘긴다. 칸(응시자·채널)마다 처리 중 1장이라 규칙 순서가 지켜진다."""
        count = 0
        for p in self.mailboxes.take_ready(self._inflight):
            slot = (p.key, p.channel)
            queued = self._queued_at.pop(slot, time.monotonic())
            frame = self._to_frame(p)
            if frame is None:
                continue
            self._inflight.add(slot)
            task = asyncio.create_task(self._infer_one(frame, slot, queued))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            count += 1
        self.metrics.inflight.set(len(self._inflight))
        return count

    async def _infer_one(self, frame: Frame, slot: tuple[str, str], queued: float) -> None:
        try:
            det = await self.detector.infer(frame)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - 모델 오류가 서버를 죽이지 않게(그 프레임만 버림)
            self.metrics.detector_errors.inc()
            log.warning("모델 실행 오류 (프레임 건너뜀): %r", exc)
            det = None
        finally:
            self._inflight.discard(slot)
            self.metrics.inflight.set(len(self._inflight))
            self._wake.set()  # 이 칸에 기다리던 프레임이 있으면 바로 넘긴다
        if det is not None:
            try:
                self._apply(det, queued, time.monotonic())
            except Exception:  # noqa: BLE001 - 규칙·콜백 오류가 태스크 밖으로 새지 않게
                log.exception("분석 결과 적용 오류")

    def _to_frame(self, p: ParsedFrame) -> Frame | None:
        session = self.registry.get(p.key)
        if session is None:
            return None
        if session.origin_ms is None and p.channel == PHONE:
            session.origin_ms = p.recv_ms
        return Frame(p.key, p.channel, p.seq, p.recv_ms, p.jpeg, session.origin_ms or p.recv_ms)

    async def process_pending(self) -> int:
        frames = self.mailboxes.take_all()
        if not frames:
            return 0
        # 한 번에 전부 넘긴다: 프로세스 풀은 응시자별 워커로 나눠 병렬 처리하고, 단일 프로세스는 안에서 차례로 처리한다
        return await self._process_batch(frames)

    async def _process_batch(self, parsed: list[ParsedFrame]) -> int:
        batch: list[Frame] = []
        queued: dict[tuple[str, str], float] = {}
        for p in parsed:
            frame = self._to_frame(p)
            if frame is None:
                continue
            # 수신 시각은 꺼낼 때 가져간다(처리 중에 같은 칸에 새 프레임이 오면 그 시각으로 덮이므로)
            queued[(p.key, p.channel)] = self._queued_at.pop((p.key, p.channel), time.monotonic())
            batch.append(frame)
        if not batch:
            return 0
        try:
            detections = await self.detector.run_async(batch)
        except Exception:  # noqa: BLE001 - 모델 오류가 서버를 죽이지 않게
            self.metrics.detector_errors.inc()
            log.exception("모델 실행 오류")
            return 0
        now = time.monotonic()
        for det in detections:
            self._apply(det, queued.get((det.key, det.channel)), now)
        return len(detections)

    def _apply(self, det: Detection, queued: float | None, now: float) -> None:
        """분석 결과 하나를 세션 규칙에 넣고 이벤트·check·지표를 낸다."""
        session = self.registry.get(det.key)
        if session is None:
            return
        out = session.on_detection(det, now)
        self.processed += 1
        self.metrics.frames_processed.labels(det.channel).inc()
        if queued is not None:
            self.metrics.frame_latency.observe(max(0.0, now - queued))
        prev = session.last_processed_recv.get(det.channel)
        if prev is not None and det.recv_ms > prev:
            self.metrics.processed_gap.labels(det.channel).observe((det.recv_ms - prev) / 1000)
        session.last_processed_recv[det.channel] = max(det.recv_ms, prev or 0)
        self.emit(det.key, out.events)
        if out.check is not None:
            self.hub.publish({"type": "check", "sessionKey": det.key, **out.check})

    def emit(self, session_key: str, events: list[RuleEvent]) -> None:
        for ev in events:
            if ev.type not in SOURCE_BY_TYPE:  # 계약에 없는 유형(예: MANY_HANDS)은 그림자 모드: 기록만
                if ev.final:
                    log.info("그림자 이벤트 %s 지속 %dms", ev.type, ev.duration_ms)
                    self.metrics.shadow_events.labels(ev.type).inc()
                continue
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
