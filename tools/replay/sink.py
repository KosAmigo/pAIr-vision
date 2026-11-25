"""백엔드 콜백을 흉내 내는 수신기: 서명·타임스탬프(±60초)를 검증하고 받은 이벤트를 출력한다.

  VISION_CALLBACK_SECRET=dev-callback-secret uv run python tools/replay/sink.py --port 9000
개발·시연용이며 DB 저장은 하지 않는다(같은 clientId 는 duplicate 로 응답하고 더 긴 지속시간으로만 갱신).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pair_vision.auth.signing import verify  # noqa: E402

SECRETS = {"k1": os.environ.get("VISION_CALLBACK_SECRET", "")}
store: dict[str, dict] = {}
app = FastAPI(docs_url=None, redoc_url=None)


@app.post("/internal/vision/events")
async def events(request: Request) -> JSONResponse:
    body = await request.body()
    ts = request.headers.get("x-vision-timestamp")
    sig = request.headers.get("x-vision-signature")
    key_id = request.headers.get("x-vision-key-id")
    if not (ts and sig and key_id):
        return JSONResponse({"code": "MISSING_SIGNATURE"}, status_code=401)
    secret = SECRETS.get(key_id)
    if not secret:
        return JSONResponse({"code": "UNKNOWN_KEY_ID"}, status_code=403)
    if not ts.isdigit() or abs(time.time() - int(ts)) > 60:
        return JSONResponse({"code": "STALE_TIMESTAMP"}, status_code=403)
    if not verify(secret, ts, body, sig):
        return JSONResponse({"code": "BAD_SIGNATURE"}, status_code=403)
    ev = json.loads(body)
    cid = ev["clientId"]
    print(f"[sink] {ev['type']} final={ev['final']} dur={ev['durationMs']}ms clientId={cid}", flush=True)
    if cid in store:
        updated = ev["durationMs"] > store[cid]["durationMs"]
        if updated:
            store[cid] = ev
        return JSONResponse({"id": 1, "duplicate": True, "updated": updated})
    store[cid] = ev
    return JSONResponse({"id": len(store), "duplicate": False}, status_code=201)


@app.get("/received")
async def received() -> list[dict]:
    return list(store.values())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000)
    uvicorn.run(app, host="0.0.0.0", port=ap.parse_args().port, access_log=False)  # noqa: S104
