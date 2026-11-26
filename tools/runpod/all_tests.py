"""총괄 시험 노트북(pair_all_tests.ipynb)의 단계 함수들. 노트북이 zip 을 풀고 나서 불러 쓴다.

각 단계는 일을 하고 한 줄 요약(str)을 돌려준다. 실패하면 예외를 던지고, 건너뛰어야 하면 Skip 을 던진다.
노트북의 `step()` 이 예외를 잡아 다음 단계로 넘어가므로 Run All 한 번에 끝난다.
결과(md·csv·json·log)는 ctx.results 아래에 모이고 `pack_results` 가 zip 하나로 묶는다.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
SCAN_NAMES = [f"h{i:02d}" for i in range(12)] + [f"v{i}" for i in range(5)] + [
    "m_front", "m_left", "m_back", "m_right", "m_floor", "m_ceiling",
]
API_KEY = "all-tests-key"
CALLBACK_SECRET = "all-tests-callback-secret"


class Skip(Exception):
    """이 단계를 건너뛴다(실패 아님)."""


@dataclass
class Ctx:
    root: Path
    quick: bool = True
    allow_cpu: bool = False
    py: str = field(default_factory=lambda: sys.executable)
    gpu: bool = False
    gpu_name: str = "cpu"
    results: Path = field(init=False)
    work: Path = field(init=False)
    extra_names: set = field(default_factory=set)  # photos/extra/ 에서 온 사진 이름(정확도를 출처별로 나누려고)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.results = self.root / "results"
        self.work = self.root.parent / "pair-vision-work"
        (self.results / "logs").mkdir(parents=True, exist_ok=True)

    @property
    def models(self) -> Path:
        return self.root / "models"

    def env(self, **extra: str) -> dict[str, str]:
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(self.root / "src"), str(self.root)]),
               "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        env.update(extra)
        return env


# --- 공통 도구 ---
def sh(ctx: Ctx, args: list[str], log: str, env: dict[str, str] | None = None, timeout: float | None = None,
       cwd: Path | None = None) -> int:
    """명령을 돌리며 출력을 화면과 results/logs/<log> 에 같이 남긴다. 종료 코드를 돌려준다."""
    path = ctx.results / "logs" / log
    print("$", " ".join(args), flush=True)
    with path.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(args, cwd=cwd or ctx.root, env=env or ctx.env(), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        started = time.monotonic()
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                f.write(line)
                print(line, end="", flush=True)
                if timeout and time.monotonic() - started > timeout:
                    proc.kill()
                    f.write(f"\n[시간 초과 {timeout:.0f}초]\n")
                    break
        finally:
            proc.wait()
    return proc.returncode


def tail(ctx: Ctx, log: str, n: int = 8) -> str:
    try:
        return " | ".join((ctx.results / "logs" / log).read_text(encoding="utf-8").strip().splitlines()[-n:])
    except OSError:
        return ""


def images_in(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def synthetic_photos(ctx: Ctx, count: int = 8) -> Path:
    """사진이 없을 때 쓰는 합성 사진(속도만 의미 있다)."""
    out = ctx.results / "synthetic-photos"
    if len(images_in(out)) >= count:
        return out
    import numpy as np
    from PIL import Image, ImageDraw

    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(count):
        image = Image.fromarray((rng.random((480, 640, 3)) * 60 + 120).astype("uint8"))
        draw = ImageDraw.Draw(image)
        for _ in range(6):
            x0, y0 = int(rng.integers(0, 500)), int(rng.integers(0, 380))
            draw.rectangle([x0, y0, x0 + int(rng.integers(40, 120)), y0 + int(rng.integers(30, 90))],
                           fill=tuple(int(v) for v in rng.integers(30, 220, 3)))
        image.save(out / f"{i + 1}.png")
    return out


PHOTO_ROOTS = ("photos", "/workspace/photos")  # 찾는 순서: zip 안 photos/ -> /workspace/photos


def photo_root(ctx: Ctx) -> Path | None:
    for r in PHOTO_ROOTS:
        base = Path(r) if Path(r).is_absolute() else ctx.root / r
        if images_in(base):
            return base
    return None


def photo_dirs(ctx: Ctx) -> dict[str, Path]:
    """phone / webcam / scan 사진 폴더. 하위 폴더(phone/ webcam/ scan/)가 없으면 폴더 전체를 휴대폰 사진으로 쓴다.

    extra/ 폴더(추가 정답 사진)는 휴대폰 사진에 합친다. 사진이 어디에도 없으면 합성 사진.
    찾은 사진은 작업 폴더(photo-set)에 모아 한 곳으로 만든다(이름이 겹치면 폴더 이름을 앞에 붙임).
    """
    cached = getattr(ctx, "_photo_dirs", None)
    if cached is not None:
        return cached
    root = photo_root(ctx)
    if root is None:
        found = {"phone": synthetic_photos(ctx)}
        ctx._photo_dirs = found  # type: ignore[attr-defined]
        return found
    kinds = {k: images_in(root / k) for k in ("phone", "webcam", "scan")}
    extra = images_in(root / "extra")
    if not any(kinds.values()):  # 하위 폴더가 없다: extra 를 뺀 전체가 휴대폰 사진
        kinds["phone"] = [p for p in images_in(root) if (root / "extra") not in p.parents]
    kinds["phone"] = kinds["phone"] + extra
    out = ctx.work / "photo-set"
    shutil.rmtree(out, ignore_errors=True)
    found = {}
    for kind, files in kinds.items():
        if not files:
            continue
        (out / kind).mkdir(parents=True, exist_ok=True)
        for p in files:
            target = out / kind / p.name
            if target.exists():
                target = out / kind / f"{p.parent.name}_{p.name}"
            shutil.copy2(p, target)
            if p in extra:
                ctx.extra_names.add(target.name)
        found[kind] = out / kind
    if "phone" not in found:
        found["phone"] = synthetic_photos(ctx)
    ctx._photo_dirs = found  # type: ignore[attr-defined]
    return found


def has_real_photos(ctx: Ctx) -> bool:
    return photo_root(ctx) is not None


def find_labels(ctx: Ctx) -> Path | None:
    roots = [ctx.root] + [Path(r) if Path(r).is_absolute() else ctx.root / r for r in PHOTO_ROOTS]
    return next((r / "labels.csv" for r in roots if (r / "labels.csv").is_file()), None)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --- ② 환경 ---
def step_env(ctx: Ctx) -> str:
    if platform.system() == "Linux" and shutil.which("bash"):
        env = ctx.env(WORK=str(ctx.work), VISION_MODELS_DIR=str(ctx.models), PYTHON=sys.executable)
        rc = sh(ctx, ["bash", str(ctx.root / "tools" / "runpod" / "bootstrap.sh")], "02_bootstrap.log", env=env, timeout=3600)
        if rc != 0:
            raise RuntimeError(f"bootstrap.sh 실패(코드 {rc}): {tail(ctx, '02_bootstrap.log')}")
        venv_py = Path(os.environ.get("VISION_VENV", ctx.work / "venv-vision")) / "bin" / "python"
        ctx.py = str(venv_py)
        detail = f"bootstrap.sh 완료, 가상환경 {venv_py}"
    else:
        detail = f"Linux 아님({platform.system()}): bootstrap.sh 를 건너뛰고 현재 파이썬({ctx.py}) 사용"
    rc = sh(ctx, [ctx.py, "-m", "pip", "install", "-q", "pytest", "pytest-asyncio", "nvidia-ml-py", "psutil", "huggingface_hub"],
            "02_pip.log", timeout=900)
    if rc != 0:
        if "No module named pip" in tail(ctx, "02_pip.log", 3):  # pip 없는 가상환경(PC 확인용): 이미 깔린 패키지로 진행
            return detail + " (pip 없음: 추가 설치 건너뜀)"
        raise RuntimeError(f"pip 설치 실패: {tail(ctx, '02_pip.log')}")
    return detail


# --- ③ GPU·모델 ---
def step_gpu_models(ctx: Ctx) -> str:
    probe = ("import json,sys,importlib.metadata as m\n"
             "out={'python':sys.version.split()[0],'pkgs':{}}\n"
             "for n in ('torch','transformers','mediapipe','numpy','pillow','fastapi'):\n"
             "    try: out['pkgs'][n]=m.version(n)\n"
             "    except Exception: out['pkgs'][n]=None\n"
             "try:\n"
             "    import torch\n"
             "    out['cuda']=bool(torch.cuda.is_available())\n"
             "    out['gpu']=torch.cuda.get_device_name(0) if out['cuda'] else ''\n"
             "    out['torch_cuda']=torch.version.cuda\n"
             "except Exception as e: out['cuda']=False; out['gpu']=''; out['err']=repr(e)\n"
             "print(json.dumps(out))\n")
    cp = subprocess.run([ctx.py, "-c", probe], capture_output=True, text=True, encoding="utf-8", errors="replace", env=ctx.env())
    try:
        info = json.loads(cp.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise RuntimeError(f"환경 조사 실패: {cp.stderr[-300:]}") from None
    ctx.gpu = bool(info["cuda"])
    ctx.gpu_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in info["gpu"]) if ctx.gpu else "cpu"
    smi = ""
    if shutil.which("nvidia-smi"):
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False).stdout
    lines = [f"날짜: {datetime.now():%Y-%m-%d %H:%M}", f"플랫폼: {platform.platform()}", f"코어: {os.cpu_count()}",
             f"python: {info['python']}", *(f"{k}: {v}" for k, v in info["pkgs"].items()),
             f"CUDA 사용 가능: {info['cuda']} {info['gpu']} (torch CUDA {info.get('torch_cuda')})", "", smi]
    (ctx.results / "env.txt").write_text("\n".join(lines), encoding="utf-8")
    rc = sh(ctx, [ctx.py, str(ctx.root / "tools" / "runpod" / "setup_models.py"), "--models-dir", str(ctx.models)],
            "03_models.log", timeout=1800)
    if rc != 0:
        raise RuntimeError(f"모델 검증 실패: {tail(ctx, '03_models.log')}")
    if not ctx.gpu and not ctx.allow_cpu:
        raise RuntimeError("GPU(CUDA)를 쓸 수 없습니다. GPU Pod 인지 확인하세요. CPU 로라도 돌리려면 ALLOW_CPU = True")
    return f"{'GPU ' + info['gpu'] if ctx.gpu else 'CPU(ALLOW_CPU)'}, 모델 5개 sha256 검증 통과"


def require_ready(ctx: Ctx) -> None:
    if not (ctx.results / "env.txt").exists():
        raise Skip("③ 단계가 먼저 성공해야 합니다")
    if not ctx.gpu and not ctx.allow_cpu:
        raise Skip("GPU 없음(ALLOW_CPU=False)")


# --- ④ pytest ---
def step_pytest(ctx: Ctx) -> str:
    env = ctx.env(VISION_MODELS_DIR=str(ctx.models))
    if has_real_photos(ctx):
        env["VISION_TEST_PHOTOS"] = str(photo_dirs(ctx)["phone"])
    rc = sh(ctx, [ctx.py, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs"], "04_pytest.log", env=env, timeout=3600)
    summary = tail(ctx, "04_pytest.log", 2)
    if rc != 0:
        raise RuntimeError(f"pytest 실패(코드 {rc}): {summary}")
    return summary


# --- ⑤ 모델 속도 비교 ---
def step_compare(ctx: Ctx) -> str:
    require_ready(ctx)
    photos = photo_dirs(ctx)["phone"]
    warmup, repeat = ("1", "1") if ctx.quick else ("5", "10")
    args = [ctx.py, "-m", "tools.bench.compare", "--images", str(photos), "--models", "dfine-s,owlv2,mediapipe-hands",
            "--hand-conf", "0.5,0.3,0.1", "--warmup", warmup, "--repeat", repeat, "--json", str(ctx.results / "compare.json")]
    if ctx.quick:
        args += ["--limit", "4"]
    rc = sh(ctx, args, "05_compare.log", env=ctx.env(VISION_MODELS_DIR=str(ctx.models)), timeout=3600)
    if rc != 0:
        raise RuntimeError(f"compare 실패(코드 {rc}): {tail(ctx, '05_compare.log')}")
    text = (ctx.results / "logs" / "05_compare.log").read_text(encoding="utf-8")
    (ctx.results / "compare.md").write_text(text, encoding="utf-8")
    return "compare.md / compare.json"


# --- ⑥ 용량 측정 ---
def step_capacity(ctx: Ctx) -> str:
    require_ready(ctx)
    dirs = photo_dirs(ctx)
    out = ctx.results / f"capacity_{ctx.gpu_name}"
    args = [ctx.py, "-u", "-m", "tools.bench.capacity", "--detector", "real", "--photos", str(dirs["phone"]),
            "--models-dir", str(ctx.models), "--out", str(out)]
    if "webcam" in dirs:
        args += ["--webcam-photos", str(dirs["webcam"])]
    if ctx.quick:
        if ctx.gpu:
            args += ["--workers", "2", "--batch", "on", "--fp16", "on", "--webcam", "server", "--examinees", "10,20", "--duration", "15"]
        else:  # CPU 는 느려서 아주 작게(스크립트가 끝까지 도는지만 본다)
            args += ["--workers", "1", "--batch", "on", "--fp16", "off", "--webcam", "server", "--examinees", "1",
                     "--duration", "6", "--warmup", "2"]
    rc = sh(ctx, args, "06_capacity.log", timeout=4 * 3600)
    if rc != 0 or not out.with_suffix(".md").exists():
        raise RuntimeError(f"capacity 실패(코드 {rc}): {tail(ctx, '06_capacity.log')}")
    md = out.with_suffix(".md").read_text(encoding="utf-8").splitlines()
    failed = [ln for ln in md if "실패:" in ln]
    if failed:  # capacity.py 는 한 조합이 실패해도 끝까지 돌고 종료 코드 0 이므로 결과 표에서 확인한다
        raise RuntimeError(f"조합 실패 {len(failed)}건: {failed[0][:200]}. 로그: logs/06_capacity.log")
    rows: list[str] = []
    for i, ln in enumerate(md):
        if ln.startswith("| workers | 배치 | fp16 | 웹캠 | 최대 인원"):  # 맨 위 요약 표
            for row in md[i + 2 :]:
                if not row.startswith("|"):
                    break
                cells = [c.strip() for c in row.strip("|").split("|")]
                rows.append(f"w{cells[0]}/배치{cells[1]}/fp16 {cells[2]}/웹캠 {cells[3]}={cells[4]}")
            break
    return f"{out.name}.md/.csv. 최대 인원: {', '.join(rows[:6])}"


# --- ⑦ 정확도 평가 ---
def _split_labels(labels: Path, names: set[str], out: Path, keep_in: bool) -> int:
    """정답 CSV 에서 names 에 있는(keep_in) 또는 없는 사진의 줄만 골라 저장한다. 고른 줄 수를 돌려준다."""
    import csv

    with labels.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if (Path(r["file"].strip()).name in names) == keep_in]
        fields = reader.fieldnames or []
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def step_eval(ctx: Ctx) -> str:
    labels = find_labels(ctx)
    if labels is None:
        raise Skip("labels.csv 없음(/workspace/photos/labels.csv 또는 번들 --labels)")
    if not (ctx.results / "compare.json").exists():
        raise Skip("⑤ compare 결과(compare.json)가 없음")
    photo_dirs(ctx)  # extra 이름 목록을 채운다
    parts = {"mine": False}
    if ctx.extra_names:
        parts = {"mine": False, "extra": True}
    done = []
    for tag, keep_in in parts.items():
        truth = ctx.results / f"labels_{tag}.csv"
        n = _split_labels(labels, ctx.extra_names, truth, keep_in)
        if n == 0:
            continue
        rc = sh(ctx, [ctx.py, "-m", "tools.eval.evaluate", "score", "--truth", str(truth), "--results",
                      str(ctx.results / "compare.json"), "--out", str(ctx.results / f"eval_{tag}")], f"07_eval_{tag}.log")
        if rc != 0:
            raise RuntimeError(f"eval({tag}) 실패(코드 {rc}): {tail(ctx, f'07_eval_{tag}.log')}")
        done.append(f"{tag} {n}장")
    if not done:
        raise Skip("정답 CSV 에 맞는 사진이 없음")
    return f"eval_*.md/csv ({', '.join(done)}), 정답 {labels}"


# --- 서버 도우미 ---
class Server:
    """실제 모델 서버를 백그라운드로 띄운다(with 문으로 쓰면 끝날 때 종료)."""

    def __init__(self, ctx: Ctx, tag: str, callback_url: str = "") -> None:
        self.ctx, self.tag, self.port = ctx, tag, free_port()
        rules = "configs/rules.default.yaml" if ctx.gpu else "configs/rules.local-cpu.yaml"
        extra = {"VISION_CALLBACK_URL": callback_url, "VISION_CALLBACK_SECRET": CALLBACK_SECRET} if callback_url else {}
        self.env = ctx.env(VISION_API_KEY=API_KEY, VISION_DETECTOR="real", VISION_MODELS_DIR=str(ctx.models),
                           VISION_PORT=str(self.port), VISION_HOST="127.0.0.1", VISION_RULES_PATH=rules,
                           VISION_MAX_SESSIONS="10", **extra)
        self.proc: subprocess.Popen | None = None
        self.url = f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> Server:
        import httpx

        log = (self.ctx.results / "logs" / f"server_{self.tag}.log").open("w", encoding="utf-8")
        self.proc = subprocess.Popen([self.ctx.py, "-m", "pair_vision"], cwd=self.ctx.root, env=self.env,
                                     stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"서버가 종료됨(코드 {self.proc.returncode}): {tail(self.ctx, f'server_{self.tag}.log')}")
            with contextlib.suppress(Exception):
                r = httpx.get(f"{self.url}/readyz", headers={"Authorization": f"Bearer {API_KEY}"}, timeout=3)
                if r.json().get("modelsLoaded"):
                    return self
            time.sleep(2)
        raise RuntimeError("서버가 15분 안에 준비되지 않았습니다")

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def jpeg_under(path: Path, limit: int = 380 * 1024, edge: int = 960) -> bytes:
    import io

    from PIL import Image

    image = Image.open(path).convert("RGB")
    image.thumbnail((edge, edge))
    for quality in (85, 70, 55, 40):
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=quality)
        if buf.tell() <= limit:
            break
    return buf.getvalue()


# --- ⑧ 방 스캔 API ---
def step_scan(ctx: Ctx) -> str:
    require_ready(ctx)
    import httpx

    dirs = photo_dirs(ctx)
    folder = dirs.get("scan") or synthetic_photos(ctx, 3)
    files = images_in(folder)[:24]
    names = SCAN_NAMES[: len(files)]  # 파일 이름이 방 스캔 이름이 아니므로 순서대로 h00, h01 ... 을 붙인다
    payload = [("images", (n, jpeg_under(p), "image/jpeg")) for n, p in zip(names, files, strict=True)]
    with Server(ctx, "scan") as srv:
        headers = {"Authorization": f"Bearer {API_KEY}"}
        calls = []
        for _ in range(2):  # 첫 호출은 워밍업 포함
            t0 = time.perf_counter()
            r = httpx.post(f"{srv.url}/v1/scan", headers=headers, files=payload, timeout=600)
            calls.append((r.status_code, round((time.perf_counter() - t0) * 1000), r.json()))
        bad = httpx.post(f"{srv.url}/v1/scan", headers=headers, files=[("images", ("zzz", b"\xff\xd8\xff", "image/jpeg"))], timeout=30)
    status, wall_ms, body = calls[-1]
    (ctx.results / "scan.json").write_text(json.dumps({"photos": len(files), "calls_ms": [c[1] for c in calls],
                                                       "response": body}, ensure_ascii=False, indent=1), encoding="utf-8")
    if status != 200 or calls[0][0] != 200:
        raise RuntimeError(f"/v1/scan 상태 {status}: {str(body)[:200]}")
    if bad.status_code != 400 or bad.json().get("code") != "BAD_SCAN":
        raise RuntimeError(f"잘못된 요청이 BAD_SCAN 이 아님: {bad.status_code} {bad.text[:100]}")
    reasons = [f"{r['code']}{r['names']}" for r in body["verdict"]["reasons"]]
    return (f"사진 {len(files)}장, verdict {body['verdict']['status']} {reasons}, 서버 {body['ms']}ms(왕복 {wall_ms}ms), "
            f"{'실제 사진' if 'scan' in dirs else '합성 사진'}, 잘못된 요청 400 BAD_SCAN 확인")


# --- ⑨ 스트림·콜백 ---
def step_replay(ctx: Ctx) -> str:
    require_ready(ctx)
    import httpx

    photos = photo_dirs(ctx)["phone"]
    sink_port = free_port()
    sink_log = (ctx.results / "logs" / "sink.log").open("w", encoding="utf-8")
    sink = subprocess.Popen([ctx.py, str(ctx.root / "tools" / "replay" / "sink.py"), "--port", str(sink_port)], cwd=ctx.root,
                            env=ctx.env(VISION_CALLBACK_SECRET=CALLBACK_SECRET), stdout=sink_log, stderr=subprocess.STDOUT)
    try:
        time.sleep(3)
        with Server(ctx, "replay", callback_url=f"http://127.0.0.1:{sink_port}/internal/vision/events") as srv:
            duration = "10" if ctx.quick else "30"
            rc = sh(ctx, [ctx.py, str(ctx.root / "tools" / "replay" / "replay.py"), "--url", srv.url, "--api-key", API_KEY,
                          "--streams", "3", "--photos", str(photos), "--hold", "4", "--max-edge", "640",
                          "--duration", duration, "--settle", "8" if not ctx.gpu else "2"], "09_replay.log", timeout=900)
        received = len(httpx.get(f"http://127.0.0.1:{sink_port}/received", timeout=5).json())
    finally:
        sink.terminate()
    log = (ctx.results / "logs" / "09_replay.log").read_text(encoding="utf-8")
    (ctx.results / "replay.md").write_text(f"```\n{log}\n```\n\n콜백 수신기가 받은 이벤트: {received}건\n", encoding="utf-8")
    if rc != 0 or "start_200" not in log:
        raise RuntimeError(f"replay 실패(코드 {rc}): {tail(ctx, '09_replay.log')}")
    return f"스트림 3개 {duration}초 재생 완료, 콜백 수신 {received}건(합성·정지 사진은 0건이 정상)"


# --- ⑩ 결과 묶기 ---
def pack_results(ctx: Ctx, summary: str) -> Path:
    (ctx.results / "summary.md").write_text(summary, encoding="utf-8")
    name = f"results_{ctx.gpu_name}_{datetime.now():%Y%m%d-%H%M}.zip"
    out = ctx.root.parent / name
    keep = (".md", ".csv", ".json", ".log", ".txt")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ctx.results.rglob("*")):
            if p.is_file() and p.suffix.lower() in keep and "synthetic-photos" not in p.parts:
                z.write(p, p.relative_to(ctx.results))
    return out
