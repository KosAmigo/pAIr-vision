"""가짜 모델 부하 시험(모델 밖 병목 실측): 응시자 N명 x 채널 2개(웹캠·휴대폰)를 같은 키로 초당 fps 장씩 보낸다.

기존 tools/replay/replay.py 는 한 번에 채널 하나만 보내서, 응시자 한 명에 두 채널을 함께 흘리려고 만든 도구다(서버 코드는 건드리지 않음).
- 웹캠 JPEG 대용 약 20KB(400px q55), 휴대폰 약 40KB(512px) 크기의 패딩. 안에 `FAKE:<STATE>` 표식을 넣어 가짜 모델이 읽는다.
- --stagger: 응시자마다 0~1/fps 초 위상차(실제 브라우저처럼 비동기). 끄면 전원이 같은 순간에 보낸다(최악 버스트).
- 실행 중 /metrics 를 --scrape-every 초마다 읽어 콜백 대기 건수 최댓값을 기록한다. 끝나면 요약 JSON 을 출력한다.
  python tools/capacity_sim/load_both.py --api-key dev-key --streams 20 --duration 60 --stagger
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pair_vision.session.frames import encode_frame, raw_to_key  # noqa: E402

PHONE_SEG = [("NORMAL", 3), ("BOTH_OUT", 10), ("NORMAL", 5), ("ONE_OUT", 12), ("NORMAL", 1)]
WEBCAM_SEG = [("FRONT", 5), ("NO_FACE", 4), ("FRONT", 6), ("HEAD_TURN", 4), ("FRONT", 3), ("MULTI_FACE", 3), ("FRONT", 1)]


def state_at(segs, t: float) -> str:
    acc = 0.0
    for s, d in segs:
        acc += d
        if t < acc:
            return s
    return segs[-1][0]


def jpeg(state: str, size: int) -> bytes:
    body = b"\xff\xd8\xff\xe0FAKE:" + state.encode()
    return body + b"\x00" * max(0, size - len(body) - 2) + b"\xff\xd9"


async def send_stream(ws, key, channel, size, segs, args, stats, offset):
    t0 = time.time() + offset
    await asyncio.sleep(max(0.0, t0 - time.time()))
    n = int(args.duration * args.fps)
    for seq in range(1, n + 1):
        now = time.time()
        await ws.send(encode_frame(key, channel, seq, int(now * 1000), jpeg(state_at(segs, now - t0), size)))
        stats[f"sent_{channel}"] += 1
        await asyncio.sleep(max(0.0, t0 + seq / args.fps - time.time()))


async def reader(ws, stats):
    async for raw in ws:
        m = json.loads(raw)
        stats[f"msg_{m.get('type')}"] += 1
        if m.get("type") == "error":
            stats[f"error_{m.get('code')}"] += 1


def parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, val = line.rpartition(" ")
        try:
            out[name] = float(val)
        except ValueError:
            pass
    return out


def cpu_seconds(pids: list[int]) -> float:
    """서버 프로세스 트리의 누적 CPU 초(윈도우 PowerShell)."""
    if not pids:
        return 0.0
    ids = ",".join(map(str, pids))
    cmd = f"(Get-Process -Id {ids} -ErrorAction SilentlyContinue | Measure-Object -Property CPU -Sum).Sum"
    out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=30).stdout.strip()
    return float(out.replace(",", ".") or 0.0)


async def scraper(http, every, peaks, stop):
    while not stop.is_set():
        try:
            m = parse_metrics((await http.get("/metrics")).text)
            peaks["callback_queue_max"] = max(peaks.get("callback_queue_max", 0), m.get("pair_vision_callback_queue_size", 0))
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(every)


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--api-key", default=os.environ.get("VISION_API_KEY", ""))
    p.add_argument("--streams", type=int, default=20)
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--webcam-bytes", type=int, default=20000)
    p.add_argument("--phone-bytes", type=int, default=40000)
    p.add_argument("--stagger", action="store_true")
    p.add_argument("--channels", default="webcam,phone")
    p.add_argument("--scrape-every", type=float, default=3.0)
    p.add_argument("--cpu-pids", default="", help="CPU 를 잴 서버 프로세스 PID 목록(쉼표)")
    p.add_argument("--one-ws", action="store_true", help="소켓 하나로 전부 보냄(기본은 응시자마다 소켓 하나)")
    args = p.parse_args()
    headers = {"Authorization": f"Bearer {args.api_key}"}
    chans = args.channels.split(",")
    keys = [raw_to_key(os.urandom(16)) for _ in range(args.streams)]
    stats: Counter = Counter()
    peaks: dict[str, float] = {}
    async with httpx.AsyncClient(base_url=args.url, headers=headers, timeout=15) as http:
        for k in keys:
            r = await http.post(f"/v1/sessions/{k}/start", json={"mode": "exam", "startedAtMs": int(time.time() * 1000)})
            stats[f"start_{r.status_code}"] += 1
        before = parse_metrics((await http.get("/metrics")).text)
        ws_url = args.url.replace("http", "ws", 1) + "/v1/stream"
        stop = asyncio.Event()
        scr = asyncio.create_task(scraper(http, args.scrape_every, peaks, stop))
        if args.one_ws:
            conns = [await websockets.connect(ws_url, additional_headers=headers, max_size=None)]
            sock = lambda i: conns[0]  # noqa: E731
        else:
            conns = list(await asyncio.gather(*[websockets.connect(ws_url, additional_headers=headers, max_size=None) for _ in keys]))
            sock = lambda i: conns[i]  # noqa: E731
        pids = [int(x) for x in args.cpu_pids.split(",") if x]
        cpu0 = await asyncio.to_thread(cpu_seconds, pids)
        started = time.time()
        readers = [asyncio.create_task(reader(c, stats)) for c in conns]
        tasks = []
        for i, k in enumerate(keys):
            for ch in chans:
                off = random.random() / args.fps if args.stagger else 0.0
                tasks.append(send_stream(sock(i), k, ch, args.webcam_bytes if ch == "webcam" else args.phone_bytes,
                                         WEBCAM_SEG if ch == "webcam" else PHONE_SEG, args, stats, off))
        await asyncio.gather(*tasks)
        send_s = time.time() - started
        cpu1 = await asyncio.to_thread(cpu_seconds, pids)
        await asyncio.sleep(3.0)
        elapsed = time.time() - started
        for r in readers:
            r.cancel()
        for c in conns:
            await c.close()
        stop.set()
        await scr
        flushed = 0
        for k in keys:
            r = await http.post(f"/v1/sessions/{k}/end")
            flushed += r.json().get("flushedEvents", 0)
        await asyncio.sleep(1.0)
        after = parse_metrics((await http.get("/metrics")).text)
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in after if k.endswith(("_total", "_sum", "_count")) or "_bucket" in k}
    summary = {
        "elapsed_s": round(elapsed, 1), "send_phase_s": round(send_s, 1),
        "server_cpu_cores": round((cpu1 - cpu0) / send_s, 3) if pids else None, "streams": args.streams, "stats": dict(stats), "flushed": flushed,
        "callback_queue_max": peaks.get("callback_queue_max", 0),
        "callback_queue_end": after.get("pair_vision_callback_queue_size"),
        "metrics_delta": {k: v for k, v in delta.items() if v and re.match(r"pair_vision_(frames|events|callback|detector|frame_latency)", k)},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
