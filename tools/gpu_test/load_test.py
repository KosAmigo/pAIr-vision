"""GPU 1장 합격 시험(docs/intent/realtime-vision-20.md): 응시자 N명 x 2채널(웹캠·휴대폰) x 초당 1장을 지정 시간 보내고
/metrics 로 합격 기준 3개를 **자동 판정**한다. 근거 표와 JSON 보고서를 낸다.

합격 기준(셋 다): 버린 사진 비율(pair_vision_frames_dropped_total / 받은 사진) <= 1% · 경고 지연 p95(pair_vision_frame_latency_seconds) <= 3초 ·
처리된 사진 간격이 끊김 허용(휴대폰 3초·웹캠 2.5초)을 넘는 비율 <= 1%(채널마다, pair_vision_processed_gap_seconds).
정합성(판정이 유효한지): /readyz gpu=true·modelsLoaded, 모델 오류 0, 콜백 실패 0 이고 수신기(sink.py, 서명 검증)가 이벤트를 받음.
처음 warmup 초는 판정에서 뺀다(모델 첫 실행·세션 보정). 지표는 warmup 끝 시점과 전송 끝 시점의 차이로 계산한다.

  python tools/gpu_test/load_test.py --api-key <키> --students 20 --duration 1200 --warmup 60 \
      --images <make_images.py 폴더> --sink-url http://127.0.0.1:9000 --out report.json
종료 코드: 0 합격, 1 불합격, 2 판정 무효(정합성 실패), 3 실행 오류.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import httpx
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pair_vision.session.frames import encode_frame, raw_to_key  # noqa: E402

GAP_LIMIT = {"phone": 3.0, "webcam": 2.5}
MAX_DROP = 0.01
MAX_P95_S = 3.0
MAX_GAP_OVER = 0.01
_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|\+Inf|-Inf)$")


# --- 지표 ---
def parse_metrics(text: str) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if m:
            out[(m.group(1), m.group(2) or "")] = float(m.group(3))
    return out


def label(labels: str, name: str) -> str | None:
    m = re.search(rf'{name}="([^"]*)"', labels)
    return m.group(1) if m else None


def total(after, before, name: str, **match) -> float:
    s = 0.0
    for (n, labels), v in after.items():
        if n == name and all(label(labels, k) == want for k, want in match.items()):
            s += v - before.get((n, labels), 0.0)
    return s


def buckets(after, before, name: str, **match) -> list[tuple[float, float]]:
    """누적 버킷 (상한, 개수) 차이, 상한 오름차순."""
    out = []
    for (n, labels), v in after.items():
        if n == name + "_bucket" and all(label(labels, k) == want for k, want in match.items()):
            le = label(labels, "le")
            out.append((math.inf if le == "+Inf" else float(le), v - before.get((n, labels), 0.0)))
    return sorted(out)


def cum_at(bs: list[tuple[float, float]], bound: float) -> float:
    return next((c for le, c in bs if abs(le - bound) < 1e-9), float("nan"))


def quantile(bs: list[tuple[float, float]], q: float) -> float | None:
    """히스토그램 분위수(버킷 안 선형 보간, prometheus histogram_quantile 과 같은 방식)."""
    if not bs or bs[-1][1] <= 0:
        return None
    rank = q * bs[-1][1]
    prev_le, prev_c = 0.0, 0.0
    for le, c in bs:
        if c >= rank:
            if math.isinf(le):
                return prev_le
            return prev_le + (le - prev_le) * ((rank - prev_c) / max(1e-9, c - prev_c))
        prev_le, prev_c = le, c
    return None


# --- 보조 측정(GPU·CPU) ---
class Sampler(threading.Thread):
    def __init__(self, every: float = 2.0) -> None:
        super().__init__(daemon=True)
        self.every, self.stop_flag = every, threading.Event()
        self.gpu: list[float] = []
        self.mem: list[float] = []
        self.load: list[float] = []

    def run(self) -> None:
        smi = shutil.which("nvidia-smi")
        while not self.stop_flag.is_set():
            if smi:
                try:
                    o = subprocess.run([smi, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0].split(",")
                    self.gpu.append(float(o[0]))
                    self.mem.append(float(o[1]))
                except Exception:  # noqa: BLE001 - 보조 측정이 시험을 멈추지 않게
                    pass
            try:
                self.load.append(os.getloadavg()[0])
            except (OSError, AttributeError):
                pass
            self.stop_flag.wait(self.every)


# --- 부하 ---
def load_images(folder: Path) -> dict[str, list[bytes]]:
    out = {}
    for ch in ("phone", "webcam"):
        files = sorted(p for p in (folder / ch).glob("*") if p.suffix.lower() in (".jpg", ".jpeg"))
        if not files:
            raise SystemExit(f"사진이 없습니다: {folder / ch} (tools/gpu_test/make_images.py 로 만드세요)")
        out[ch] = [p.read_bytes() for p in files]
    probe = folder / "probe" / "probe.jpg"
    out["probe"] = [probe.read_bytes()] if probe.is_file() else []
    return out


async def sender(ws, key, channel, images, deadline, fps, stats, rng):
    await asyncio.sleep(rng.random() / fps)  # 응시자·채널마다 위상 무작위(실제 브라우저처럼)
    seq, t0 = 0, time.time()
    while time.time() < deadline:
        seq += 1
        data = images[(seq + hash(key)) % len(images)]
        try:
            await ws.send(encode_frame(key, channel, seq, int(time.time() * 1000), data))
            stats[f"sent_{channel}"] += 1
        except Exception as exc:  # noqa: BLE001
            stats[f"send_error_{type(exc).__name__}"] += 1
            return
        await asyncio.sleep(max(0.0, t0 + seq / fps - time.time()))


async def reader(ws, stats):
    try:
        async for raw in ws:
            if isinstance(raw, str):
                m = json.loads(raw)
                stats[f"msg_{m.get('type')}"] += 1
                if m.get("type") == "error":
                    stats[f"error_{m.get('code')}"] += 1
    except Exception:  # noqa: BLE001 - 연결 종료
        pass


async def run(args) -> dict:
    headers = {"Authorization": f"Bearer {args.api_key}"}
    images = load_images(Path(args.images))
    keys = [raw_to_key(os.urandom(16)) for _ in range(args.students)]
    stats: Counter = Counter()
    rng = random.Random(args.seed)
    async with httpx.AsyncClient(base_url=args.url, headers=headers, timeout=30) as http:
        ready = (await http.get("/readyz")).json()
        for k in keys:
            r = await http.post(f"/v1/sessions/{k}/start", json={"mode": "exam", "startedAtMs": int(time.time() * 1000)})
            stats[f"start_{r.status_code}"] += 1
        ws_url = args.url.replace("http", "ws", 1) + "/v1/stream"
        conns = await asyncio.gather(*[websockets.connect(ws_url, additional_headers=headers, max_size=None) for _ in keys])
        readers = [asyncio.create_task(reader(c, stats)) for c in conns]
        start = time.time()
        deadline = start + args.warmup + args.duration
        # 응시자 0 의 휴대폰은 손 없는 책상(probe)을 보내 손 이탈 이벤트 -> 서명 콜백이 반드시 생기게 한다(인원·부하는 같다)
        pick = lambda i, ch: images["probe"] if (i == 0 and ch == "phone" and images["probe"]) else images[ch]  # noqa: E731
        tasks = [asyncio.create_task(sender(conns[i], k, ch, pick(i, ch), deadline, args.fps, stats, rng))
                 for i, k in enumerate(keys) for ch in ("phone", "webcam")]
        sampler = Sampler()
        before = None
        last_print = start
        while time.time() < deadline:
            await asyncio.sleep(1.0)
            now = time.time()
            if before is None and now - start >= args.warmup:
                before = parse_metrics((await http.get("/metrics")).text)
                sampler.start()
                print(f"[{now - start:6.0f}s] 준비 구간 끝 - 지금부터 {args.duration:.0f}초 판정 측정", flush=True)
            if now - last_print >= args.progress_every:
                last_print = now
                m = parse_metrics((await http.get("/metrics")).text)
                ref = before or {}
                recv = total(m, ref, "pair_vision_frames_received_total")
                drop = total(m, ref, "pair_vision_frames_dropped_total")
                lat = quantile(buckets(m, ref, "pair_vision_frame_latency_seconds"), 0.95)
                print(f"[{now - start:6.0f}s] 받음 {recv:.0f} 버림 {drop:.0f} ({drop / max(1, recv):.1%}) 지연p95 "
                      f"{(lat or 0):.2f}s 처리중 {m.get(('pair_vision_inflight_frames', ''), 0):.0f} "
                      f"GPU {sampler.gpu[-1] if sampler.gpu else '-'}%", flush=True)
        await asyncio.gather(*tasks)
        measured_s = time.time() - start - args.warmup
        await asyncio.sleep(3.0)  # 처리 중이던 프레임이 끝나도록
        after = parse_metrics((await http.get("/metrics")).text)
        sampler.stop_flag.set()
        for c in conns:
            await c.close()
        for r in readers:
            r.cancel()
        flushed = 0
        for k in keys:
            flushed += (await http.post(f"/v1/sessions/{k}/end")).json().get("flushedEvents", 0)
        await asyncio.sleep(4.0)  # 콜백 전송 대기
        final = parse_metrics((await http.get("/metrics")).text)
        sink_received = None
        if args.sink_url:
            try:
                sink_received = len(httpx.get(args.sink_url.rstrip("/") + "/received", timeout=10).json())
            except Exception as exc:  # noqa: BLE001
                stats[f"sink_error_{type(exc).__name__}"] += 1
    return judge(args, ready, before or {}, after, final, measured_s, sampler, stats, flushed, sink_received)


def judge(args, ready, before, after, final, measured_s, sampler, stats, flushed, sink_received) -> dict:
    received = total(after, before, "pair_vision_frames_received_total")
    processed = total(after, before, "pair_vision_frames_processed_total")
    drops = {label(lb, "reason"): v - before.get((n, lb), 0.0) for (n, lb), v in after.items()
             if n == "pair_vision_frames_dropped_total"}
    dropped = sum(drops.values())
    lat_b = buckets(after, before, "pair_vision_frame_latency_seconds")
    lat_count = lat_b[-1][1] if lat_b else 0
    within3 = cum_at(lat_b, MAX_P95_S) / lat_count if lat_count else 0.0
    gap = {}
    for ch, lim in GAP_LIMIT.items():
        gb = buckets(after, before, "pair_vision_processed_gap_seconds", channel=ch)
        cnt = gb[-1][1] if gb else 0
        over = (cnt - cum_at(gb, lim)) / cnt if cnt else 1.0
        gap[ch] = {"count": int(cnt), "over_ratio": over, "p95_s": quantile(gb, 0.95), "limit_s": lim}
    stages = {}
    for (n, lb), v in after.items():
        if n != "pair_vision_stage_seconds_total":
            continue
        st = label(lb, "stage")
        secs = v - before.get((n, lb), 0.0)
        calls = total(after, before, "pair_vision_stage_calls_total", stage=st)
        bsum = total(after, before, "pair_vision_batch_size_sum", stage=st)
        bcnt = total(after, before, "pair_vision_batch_size_count", stage=st)
        if calls > 0:
            stages[st] = {"frames": int(calls), "ms_per_frame": round(secs / calls * 1000, 2),
                          "busy": round(secs / max(1e-9, measured_s), 3),
                          "batches": int(bcnt), "avg_batch": round(bsum / bcnt, 2) if bcnt else None}
    events = total(final, before, "pair_vision_events_total")
    cb_fail = total(final, before, "pair_vision_callback_failures_total")
    cb_drop = total(final, before, "pair_vision_callback_dropped_total")
    det_err = total(after, before, "pair_vision_detector_errors_total")
    crit = {
        "drop": {"value": dropped / received if received else 1.0, "limit": MAX_DROP},
        "latency_p95": {"value": quantile(lat_b, 0.95), "within_3s_ratio": within3, "limit": MAX_P95_S},
        "gap_over": {"value": max(g["over_ratio"] for g in gap.values()), "limit": MAX_GAP_OVER, "by_channel": gap},
    }
    crit["drop"]["ok"] = received > 0 and crit["drop"]["value"] <= MAX_DROP
    crit["latency_p95"]["ok"] = lat_count > 0 and within3 >= 0.95  # 버킷 경계 3초에서 정확히: 95% 이상이 3초 안
    crit["gap_over"]["ok"] = all(g["count"] > 0 and g["over_ratio"] <= MAX_GAP_OVER for g in gap.values())
    sane = {
        "gpu": bool(ready.get("gpu")),
        "models_loaded": bool(ready.get("modelsLoaded")),
        "detector_errors_zero": det_err == 0,
        # 콜백 실패·버림 0. --require-events(짧은 확인)면 이벤트가 실제로 나서 서명 검증 수신기에 도착해야 한다(응시자 0 의 probe
        # 사진이 손 이탈을 만든다). 본 시험은 과부하면 이벤트가 안 날 수 있어(끊김) 그것을 '무효'가 아니라 기준 실패로 본다.
        "callback_ok": cb_fail == 0 and cb_drop == 0 and (
            (events > 0 and (sink_received is None or sink_received > 0)) if args.require_events
            else (events == 0 or sink_received is None or sink_received > 0)),
        "all_frames_accounted": received > 0 and processed > 0,
    }
    if args.allow_cpu:
        sane["gpu"] = True  # CPU 흐름 확인용(판정 수치는 의미 없음)
    passed = all(c["ok"] for c in crit.values())
    valid = all(sane.values())
    verdict = ("합격" if passed else "불합격") if valid else "판정 무효(정합성 실패)"
    return {
        "verdict": verdict, "passed": passed, "valid": valid, "students": args.students, "fps": args.fps,
        "measured_s": round(measured_s, 1), "warmup_s": args.warmup, "readyz": ready,
        "criteria": crit, "sanity": sane,
        "counts": {"received": int(received), "processed": int(processed), "dropped": {k: int(v) for k, v in drops.items() if v},
                   "events": int(events), "flushed_at_end": flushed, "callback_failures": int(cb_fail),
                   "callback_dropped": int(cb_drop), "sink_received": sink_received, "detector_errors": int(det_err)},
        "stages": stages,
        "gpu_util": {"avg": round(sum(sampler.gpu) / len(sampler.gpu), 1) if sampler.gpu else None,
                     "max": max(sampler.gpu) if sampler.gpu else None,
                     "mem_max_mib": max(sampler.mem) if sampler.mem else None},
        "load1_avg": round(sum(sampler.load) / len(sampler.load), 2) if sampler.load else None,
        "client": dict(stats),
    }


def fmt_pct(v) -> str:
    return "-" if v is None else f"{v:.2%}"


def print_report(rep: dict) -> None:
    c = rep["criteria"]
    ok = lambda b: "통과" if b else "실패"  # noqa: E731
    print("\n" + "=" * 24 + f" 판정: {rep['verdict']} " + "=" * 24)
    print(f"응시자 {rep['students']}명 x 2채널 x 초당 {rep['fps']}장, 판정 구간 {rep['measured_s']}초(앞 {rep['warmup_s']}초 제외)")
    print("| 기준 | 측정값 | 한도 | 결과 |\n|---|---|---|---|")
    print(f"| 버린 사진 비율 | {fmt_pct(c['drop']['value'])} | 1% | {ok(c['drop']['ok'])} |")
    lat = c["latency_p95"]
    print(f"| 경고 지연 p95 | {lat['value'] if lat['value'] is None else round(lat['value'], 2)}초 "
          f"(3초 안 {fmt_pct(lat['within_3s_ratio'])}) | 3초 | {ok(lat['ok'])} |")
    for ch, g in c["gap_over"]["by_channel"].items():
        print(f"| 끊김 초과({ch}, >{g['limit_s']}초) | {fmt_pct(g['over_ratio'])} (간격 p95 {g['p95_s'] and round(g['p95_s'], 2)}초, "
              f"{g['count']}건) | 1% | {ok(g['over_ratio'] <= MAX_GAP_OVER and g['count'] > 0)} |")
    print("\n정합성:", ", ".join(f"{k}={'OK' if v else '실패'}" for k, v in rep["sanity"].items()))
    n = rep["counts"]
    print(f"받음 {n['received']} · 처리 {n['processed']} · 버림 {n['dropped']} · 이벤트 {n['events']} · 콜백 실패 {n['callback_failures']} · "
          f"수신기 도착(서명 검증) {n['sink_received']} · 모델 오류 {n['detector_errors']}")
    g = rep["gpu_util"]
    print(f"GPU 사용률 평균 {g['avg']}% / 최대 {g['max']}% · GPU 메모리 최대 {g['mem_max_mib']}MiB · load1 평균 {rep['load1_avg']}")
    print("\n| 단계 | 프레임 | 프레임당 ms | 점유(초/초) | 묶음 수 | 평균 묶음 |\n|---|---|---|---|---|---|")
    for st, v in sorted(rep["stages"].items(), key=lambda kv: -kv[1]["busy"]):
        print(f"| {st} | {v['frames']} | {v['ms_per_frame']} | {v['busy']} | {v['batches']} | {v['avg_batch']} |")
    gpu_busy = sum(v["busy"] for k, v in rep["stages"].items() if k in ("dfine", "owlv2", "rtmw", "person"))
    print(f"(GPU 단계 점유 합 {gpu_busy:.2f}: 1.0 에 가까우면 엔진 스레드(GPU)가 꽉 찬 것. prepare·decode 는 CPU 전처리 스레드)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--api-key", default=os.environ.get("VISION_API_KEY", ""))
    ap.add_argument("--students", type=int, default=20)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=1200.0, help="판정 구간(초). 합격 시험은 1200 이상")
    ap.add_argument("--warmup", type=float, default=60.0)
    ap.add_argument("--images", required=True)
    ap.add_argument("--sink-url", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--progress-every", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--require-events", action="store_true", help="이벤트가 나서 서명 콜백이 도착해야 정합성 통과(짧은 확인용)")
    ap.add_argument("--allow-cpu", action="store_true", help="GPU 가 없어도 판정을 무효로 하지 않음(흐름 확인용)")
    args = ap.parse_args()
    try:
        rep = asyncio.run(run(args))
    except Exception as exc:  # noqa: BLE001
        print(f"부하 시험 실행 오류: {exc!r} (서버 주소·키·서버 로그 확인)", file=sys.stderr)
        return 3
    print_report(rep)
    if args.out:
        Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print("보고서:", args.out)
    return 0 if rep["passed"] and rep["valid"] else (2 if not rep["valid"] else 1)


if __name__ == "__main__":
    raise SystemExit(main())
