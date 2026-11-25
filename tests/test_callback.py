import json

import httpx
import pytest

from pair_vision.auth.signing import verify
from pair_vision.evidence.callback import CallbackSender, retry_delay

URL = "http://backend.test/internal/vision/events"


class Clock:
    def __init__(self, t=1764000000.0):
        self.t = t

    def __call__(self):
        return self.t


def make_sender(statuses, max_queue=1000):
    """statuses: 순서대로 돌려줄 상태 코드(또는 예외). 요청은 calls 에 모은다."""
    calls = []
    it = iter(statuses)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        s = next(it, 200)
        if isinstance(s, Exception):
            raise s
        return httpx.Response(s, json={})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    clock = Clock()
    return CallbackSender(URL, "k1", "sec", client, clock=clock, max_queue=max_queue), calls, clock


def event(cid="v1", duration=7000, final=False):
    return {"clientId": cid, "type": "HANDS_OUT", "durationMs": duration, "final": final}


def test_retry_schedule():
    assert [retry_delay(n) for n in range(1, 9)] == [1, 2, 4, 8, 16, 30, 30, 30]


async def test_request_is_signed_per_contract():
    sender, calls, _ = make_sender([201])
    sender.enqueue(event())
    await sender.deliver_due()
    (req,) = calls
    assert req.headers["x-vision-key-id"] == "k1"
    ts = req.headers["x-vision-timestamp"]
    assert ts == "1764000000"
    assert verify("sec", ts, req.content, req.headers["x-vision-signature"])
    assert json.loads(req.content)["clientId"] == "v1"
    assert sender.pending == 0


@pytest.mark.parametrize("status", [400, 404, 409, 422, 413])
async def test_no_retry_statuses_are_dropped(status):
    sender, calls, clock = make_sender([status])
    sender.enqueue(event())
    await sender.deliver_due()
    assert sender.pending == 0
    clock.t += 100
    await sender.deliver_due()
    assert len(calls) == 1


@pytest.mark.parametrize("status", [429, 500, 503, 403])
async def test_retry_statuses_back_off_1_2_4(status):
    sender, calls, clock = make_sender([status, status, status, 201])
    sender.enqueue(event())
    await sender.deliver_due()  # 실패 1 -> 1초 뒤
    assert sender.pending == 1 and len(calls) == 1
    clock.t += 0.5
    await sender.deliver_due()
    assert len(calls) == 1
    clock.t += 0.5
    await sender.deliver_due()  # 실패 2 -> 2초 뒤
    clock.t += 2
    await sender.deliver_due()  # 실패 3 -> 4초 뒤
    clock.t += 4
    await sender.deliver_due()  # 성공
    assert len(calls) == 4 and sender.pending == 0
    assert len({c.content for c in calls}) == 1  # 같은 본문
    assert calls[0].headers["x-vision-timestamp"] != calls[3].headers["x-vision-timestamp"]


async def test_connection_error_retries():
    sender, calls, clock = make_sender([httpx.ConnectError("x"), 200])
    sender.enqueue(event())
    await sender.deliver_due()
    assert sender.pending == 1
    clock.t += 1
    await sender.deliver_due()
    assert sender.pending == 0 and len(calls) == 2


async def test_gives_up_after_ten_minutes():
    sender, _, clock = make_sender([503] * 100)
    sender.enqueue(event())
    for _ in range(100):
        await sender.deliver_due()
        if sender.pending == 0:
            break
        clock.t += 30
    assert sender.pending == 0
    assert clock.t - 1764000000 <= 660


def test_queue_drops_oldest_beyond_limit():
    sender, _, _ = make_sender([], max_queue=3)
    for i in range(5):
        sender.enqueue(event(cid=f"v{i}"))
    assert sender.pending == 3
    assert sorted(p.client_id for p in sender._pending.values()) == ["v2", "v3", "v4"]


async def test_final_event_replaces_pending_body():
    sender, calls, _ = make_sender([200])
    sender.enqueue(event(duration=7000))
    sender.enqueue(event(duration=9000, final=True))
    assert sender.pending == 1
    await sender.deliver_due()
    body = json.loads(calls[0].content)
    assert body["durationMs"] == 9000 and body["final"] is True
