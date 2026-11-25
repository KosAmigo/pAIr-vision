"""WebSocket /v1/stream: 백엔드 -> 추론 서버 프레임(바이너리), 추론 서버 -> 백엔드 state·check·error(텍스트)."""

from __future__ import annotations

import asyncio
import contextlib
import json

from fastapi import APIRouter, WebSocket

router = APIRouter()


async def _pump(ws: WebSocket, queue: asyncio.Queue[dict]) -> None:
    while True:
        await ws.send_text(json.dumps(await queue.get(), separators=(",", ":")))


@router.websocket("/v1/stream")
async def stream(ws: WebSocket) -> None:
    core = ws.app.state.core
    if not core.verifier.verify(ws.headers.get("authorization")):
        await ws.close(code=1008)  # 핸드셰이크 거절
        return
    await ws.accept()
    hub_queue = core.hub.subscribe()
    own_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=1000)  # 이 연결로만 가는 오류
    tasks = [asyncio.create_task(_pump(ws, hub_queue)), asyncio.create_task(_pump(ws, own_queue))]

    def reply(message: dict) -> None:
        if not own_queue.full():
            own_queue.put_nowait(message)

    try:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                break
            data = message.get("bytes")
            if data is not None:
                core.pipeline.ingest(data, reply)
            # 텍스트 메시지는 아직 정의된 것이 없어 무시한다
    finally:
        core.hub.unsubscribe(hub_queue)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
