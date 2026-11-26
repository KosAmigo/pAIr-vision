"""서버 주소로 /healthz 와 WebSocket(/v1/stream) 이 닿는지 확인한다(RunPod 프록시가 WebSocket 을 통과시키는지 보려고).

  python tools/runpod/check_proxy.py <서버 주소 https://...> <API 키>
"""

from __future__ import annotations

import asyncio
import sys
import urllib.request


async def ws_check(base: str, key: str) -> str:
    import websockets

    url = base.replace("https://", "wss://").replace("http://", "ws://") + "/v1/stream"
    try:
        async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {key}"}, open_timeout=10):
            return "WebSocket 연결 성공"
    except Exception as exc:  # noqa: BLE001 - 진단 출력용
        return f"WebSocket 연결 실패: {exc!r}"


def main() -> int:
    base, key = sys.argv[1].rstrip("/"), sys.argv[2]
    try:
        print("/healthz ->", urllib.request.urlopen(base + "/healthz", timeout=10).read().decode())  # noqa: S310
    except Exception as exc:  # noqa: BLE001
        print("/healthz 실패:", exc, "(8000 포트 공개·서버 실행 여부 확인)")
        return 1
    result = asyncio.run(ws_check(base, key))
    print(result)
    return 0 if result.endswith("성공") else 1


if __name__ == "__main__":
    raise SystemExit(main())
