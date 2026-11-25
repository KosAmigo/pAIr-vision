"""백엔드 `POST /internal/vision/events` 클라이언트 (계약 §3-4).

- 서명 = hex(HMAC-SHA256(비밀, timestamp + "." + 본문)), 시도마다 타임스탬프·서명만 새로 만든다.
- 재시도 1·2·4·8·16·30초 뒤 30초 간격, 첫 대기열 진입 후 10분까지. 대기 큐 최대 1000건(오래된 것부터 버림).
- 400/404/409(+413·422 는 같은 본문을 다시 보내도 소용없어) 는 재시도 없이 버린다.
- 같은 clientId 가 아직 대기 중이면 본문을 최신(final)으로 바꿔 한 번만 보낸다.
- 본문·서명·키는 로그에 남기지 않는다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import httpx

from pair_vision.auth.signing import sign
from pair_vision.metrics import Metrics

log = logging.getLogger("pair_vision.callback")

RETRY_DELAYS_S = (1, 2, 4, 8, 16, 30)
STEADY_DELAY_S = 30
MAX_RETRY_WINDOW_S = 600
MAX_QUEUE = 1000
NO_RETRY_STATUS = frozenset({400, 404, 409, 413, 422})


def retry_delay(attempts_done: int) -> int:
    """attempts_done 번 실패한 뒤 기다릴 초."""
    if attempts_done <= 0:
        return 0
    return RETRY_DELAYS_S[attempts_done - 1] if attempts_done <= len(RETRY_DELAYS_S) else STEADY_DELAY_S


class EventSink(Protocol):
    def enqueue(self, event: dict) -> None: ...


class NullSender:
    """VISION_CALLBACK_URL 이 없을 때: 이벤트를 보내지 않고 버린 수만 센다."""

    def __init__(self, metrics: Metrics | None = None) -> None:
        self._metrics = metrics

    def enqueue(self, event: dict) -> None:
        if self._metrics:
            self._metrics.callback_dropped.labels("no_callback_url").inc()


@dataclass
class _Pending:
    seq: int
    client_id: str
    event_type: str
    body: bytes
    first_at: float
    attempts: int = 0
    next_at: float = 0.0


def _serialize(event: dict) -> bytes:
    return json.dumps(event, separators=(",", ":"), ensure_ascii=False).encode()


class CallbackSender:
    def __init__(
        self,
        url: str,
        key_id: str,
        secret: str,
        client: httpx.AsyncClient,
        *,
        clock: Callable[[], float] = time.time,
        max_queue: int = MAX_QUEUE,
        metrics: Metrics | None = None,
    ) -> None:
        self._url, self._key_id, self._secret = url, key_id, secret
        self._client = client
        self._clock = clock
        self._max_queue = max_queue
        self._metrics = metrics
        self._pending: dict[int, _Pending] = {}
        self._by_client: dict[str, int] = {}
        self._inflight: int | None = None
        self._next_seq = 0
        self._stop = asyncio.Event()

    @property
    def pending(self) -> int:
        return len(self._pending)

    def _gauge(self) -> None:
        if self._metrics:
            self._metrics.callback_queue.set(len(self._pending))

    def _count_drop(self, reason: str) -> None:
        if self._metrics:
            self._metrics.callback_dropped.labels(reason).inc()

    def _remove(self, item: _Pending) -> None:
        self._pending.pop(item.seq, None)
        if self._by_client.get(item.client_id) == item.seq:
            del self._by_client[item.client_id]
        self._gauge()

    def enqueue(self, event: dict) -> None:
        body = _serialize(event)
        client_id = event["clientId"]
        seq = self._by_client.get(client_id)
        if seq is not None and seq != self._inflight and seq in self._pending:
            self._pending[seq].body = body  # 대기 중인 같은 이벤트: 최신 본문으로 교체
            return
        while len(self._pending) >= self._max_queue:
            oldest = min((s for s in self._pending if s != self._inflight), default=None)
            if oldest is None:
                break
            self._remove(self._pending[oldest])
            self._count_drop("queue_full")
        now = self._clock()
        item = _Pending(self._next_seq, client_id, event.get("type", "?"), body, now, next_at=now)
        self._next_seq += 1
        self._pending[item.seq] = item
        self._by_client[client_id] = item.seq
        self._gauge()

    async def deliver_due(self) -> int:
        """지금 보낼 차례인 항목을 한 번씩 시도한다. 시도한 개수를 돌려준다."""
        now = self._clock()
        due = [p for p in self._pending.values() if p.next_at <= now]
        for item in due:
            if item.seq not in self._pending:
                continue
            self._inflight = item.seq
            try:
                await self._attempt(item)
            finally:
                self._inflight = None
        return len(due)

    async def _attempt(self, item: _Pending) -> None:
        timestamp = str(int(self._clock()))
        headers = {
            "Content-Type": "application/json",
            "X-Vision-Key-Id": self._key_id,
            "X-Vision-Timestamp": timestamp,
            "X-Vision-Signature": sign(self._secret, timestamp, item.body),
        }
        reason: str
        try:
            resp = await self._client.post(self._url, content=item.body, headers=headers)
        except httpx.HTTPError as exc:
            reason = f"conn:{type(exc).__name__}"
        else:
            if resp.status_code in (200, 201):
                self._remove(item)
                return
            if resp.status_code in NO_RETRY_STATUS:
                log.warning("콜백 거절(재시도 안 함) status=%s type=%s", resp.status_code, item.event_type)
                self._failure(f"http_{resp.status_code}")
                self._count_drop(f"rejected_{resp.status_code}")
                self._remove(item)
                return
            reason = f"http_{resp.status_code}"
        self._failure(reason)
        self._schedule_retry(item, reason)

    def _failure(self, reason: str) -> None:
        if self._metrics:
            self._metrics.callback_failures.labels(reason).inc()

    def _schedule_retry(self, item: _Pending, reason: str) -> None:
        item.attempts += 1
        now = self._clock()
        delay = retry_delay(item.attempts)
        if now + delay - item.first_at > MAX_RETRY_WINDOW_S:
            log.error("콜백 재시도 시간 초과, 버림 type=%s reason=%s", item.event_type, reason)
            self._count_drop("expired")
            self._remove(item)
            return
        item.next_at = now + delay
        log.warning("콜백 실패, %ss 뒤 재시도 type=%s reason=%s", delay, item.event_type, reason)

    async def run(self, interval_s: float = 0.25) -> None:
        while not self._stop.is_set():
            try:
                await self.deliver_due()
            except Exception:  # noqa: BLE001 - 전송 루프는 죽지 않는다
                log.exception("콜백 전송 루프 오류")
            await asyncio.sleep(interval_s)

    async def close(self) -> None:
        self._stop.set()
        await self._client.aclose()
