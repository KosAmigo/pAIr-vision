"""가짜 백엔드 클라이언트: N개 스트림을 start 하고 시나리오 프레임을 초당 fps 로 보낸다.

사용 예 (서버가 떠 있을 때):
  uv run python tools/replay/replay.py --streams 60 --scenario hands_out --duration 20

시나리오 이름은 VISION_FAKE_SCENARIO 와 같은 형식(`NORMAL:3,BOTH_OUT:10,NORMAL:1` 도 가능)이며,
프레임 JPEG 안에 `FAKE:<STATE>` 표식을 넣어 서버의 가짜 모델이 그 상태로 읽게 한다.

실제 모델(VISION_DETECTOR=real) 시험: `--photos <폴더>` 로 실제 사진(png/jpg)을 JPEG 로 바꿔 흘린다.
  python tools/replay/replay.py --api-key dev-key --photos C:/dev/bench-photos --photo-names 1.png --hold 20 --duration 20
사진은 `--hold` 초씩 돌아가며 보낸다(--photo-names 가 없으면 폴더 전체, 이름순). 사진은 git 에 올리지 않는다.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import httpx
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pair_vision.models.fake import parse_scenario, state_at  # noqa: E402
from pair_vision.session.frames import encode_frame, raw_to_key  # noqa: E402


def make_jpeg(state: str, pad: int) -> bytes:
    return b"\xff\xd8\xff\xe0FAKE:" + state.encode() + b"\x00" * pad + b"\xff\xd9"


def load_photos(folder: str, names: str, quality: int, max_edge: int = 0) -> list[tuple[str, bytes]]:
    """사진 폴더 -> (이름, JPEG 바이트). PNG 는 JPEG 로 다시 인코딩한다(서버는 JPEG 만 받는다)."""
    import io

    from PIL import Image

    root = Path(folder)
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    paths = [root / n for n in wanted] or sorted(
        p for p in root.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not paths:
        sys.exit(f"사진이 없습니다: {folder}")
    photos = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        if max_edge:
            img.thumbnail((max_edge, max_edge))  # 계약의 rate.maxEdge(웹캠 400, 휴대폰 512)처럼 줄인다
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
        photos.append((path.name, buf.getvalue()))
    return photos


async def stream_one(ws, key: str, segments, args, stats: Counter, photos=None) -> None:
    t0 = time.time()
    for seq in range(1, int(args.duration * args.fps) + 1):
        now = time.time()
        if photos:
            jpeg = photos[int((now - t0) / args.hold) % len(photos)][1]
        else:
            jpeg = make_jpeg(state_at(segments, now - t0), args.pad)
        await ws.send(encode_frame(key, args.channel, seq, int(now * 1000), jpeg))
        stats["frames_sent"] += 1
        await asyncio.sleep(max(0.0, t0 + seq / args.fps - time.time()))


async def reader(ws, stats: Counter) -> None:
    async for raw in ws:
        msg = json.loads(raw)
        stats[f"msg_{msg.get('type')}"] += 1
        if msg.get("type") == "error":
            stats[f"error_{msg.get('code')}"] += 1


async def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--api-key", default=os.environ.get("VISION_API_KEY", ""))
    p.add_argument("--streams", type=int, default=1)
    p.add_argument("--scenario", default="hands_out")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--duration", type=float, default=20.0, help="초")
    p.add_argument("--pad", type=int, default=14000, help="JPEG 대용 패딩 바이트(장당 약 14KB)")
    p.add_argument("--photos", default="", help="실제 사진 폴더(지정하면 가짜 표식 대신 이 사진을 보냄)")
    p.add_argument("--photo-names", default="", help="쉼표로 나눈 사진 이름(기본: 폴더 전체)")
    p.add_argument("--hold", type=float, default=8.0, help="사진 한 장을 이 초 동안 보냄")
    p.add_argument("--jpeg-quality", type=int, default=85)
    p.add_argument("--max-edge", type=int, default=0, help="긴 변을 이 픽셀로 줄임(0=그대로). 계약: 웹캠 400·품질 0.55")
    p.add_argument("--settle", type=float, default=0.0, help="마지막 프레임 뒤 추가 대기 초(느린 CPU 모델용)")
    p.add_argument("--channel", default="phone", choices=["phone", "webcam"], help="프레임 채널(webcam 은 얼굴·포즈 분석)")
    p.add_argument("--mode", default="exam", choices=["exam", "check"])
    args = p.parse_args()
    if not args.api_key:
        sys.exit("--api-key 또는 VISION_API_KEY 가 필요합니다.")

    headers = {"Authorization": f"Bearer {args.api_key}"}
    segments = parse_scenario(args.scenario) if not args.photos else []
    photos = load_photos(args.photos, args.photo_names, args.jpeg_quality, args.max_edge) if args.photos else None
    keys = [raw_to_key(os.urandom(16)) for _ in range(args.streams)]
    stats: Counter = Counter()
    async with httpx.AsyncClient(base_url=args.url, headers=headers, timeout=10) as http:
        ready = (await http.get("/readyz")).json()
        print("readyz:", {k: ready.get(k) for k in ("ready", "modelsLoaded", "gpu", "rulesVersion")})
        for key in keys:
            r = await http.post(
                f"/v1/sessions/{key}/start",
                json={"mode": args.mode, "startedAtMs": int(time.time() * 1000)},
            )
            stats[f"start_{r.status_code}"] += 1
        ws_url = args.url.replace("http", "ws", 1) + "/v1/stream"
        async with websockets.connect(ws_url, additional_headers=headers, max_size=None) as ws:
            read_task = asyncio.create_task(reader(ws, stats))
            started = time.time()
            await asyncio.gather(*(stream_one(ws, k, segments, args, stats, photos) for k in keys))
            await asyncio.sleep(2.5 + args.settle)  # 마지막 state 메시지를 받을 시간
            read_task.cancel()
            elapsed = time.time() - started
        flushed = 0
        for key in keys:
            r = await http.post(f"/v1/sessions/{key}/end")
            flushed += r.json().get("flushedEvents", 0)
        metrics = (await http.get("/metrics")).text
    print(f"스트림 {args.streams}개, {elapsed:.1f}초, 보낸 프레임 {stats['frames_sent']}장")
    print("응답 집계:", dict(stats))
    print("end 로 마무리된 이벤트:", flushed)
    for line in metrics.splitlines():
        if line.startswith(("pair_vision_frames_dropped_total", "pair_vision_frames_processed_total",
                            "pair_vision_events_total", "pair_vision_callback")) and not line.startswith("#"):
            print(" ", line)
    _ = statistics  # 지연 통계는 /metrics 의 frame_latency 히스토그램으로 본다
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
