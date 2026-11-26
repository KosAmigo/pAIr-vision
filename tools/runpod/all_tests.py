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
import re
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
    detector: str = "real"  # 용량·진단에 쓸 detector(PC 에서 도구 확인할 때만 fake)
    diag: dict | None = None  # ⑥-0 진단이 고른 설정(⑥ 용량이 이어 쓴다)

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


def _writable(folder: Path) -> bool:
    """쓰기를 실제로 시험한다(/workspace 는 권한이 애매하다)."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / ".write-test"
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except OSError:
        return False


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def step_extra(ctx: Ctx, url: str = "", size: int = 0, sha: str = "") -> str:
    """공개 정답 사진(extra.zip)을 구글 드라이브에서 받아 사진 폴더의 extra/ 로 푼다.

    어느 사진 폴더에든 extra/labels.csv 가 이미 있으면 받지 않는다. 풀 곳은 /workspace/photos(Pod 를 꺼도 남음)를
    먼저 쓰고, 쓰기가 안 되면 zip 안 photos/ 로 대체한다.
    """
    roots = [Path(r) if Path(r).is_absolute() else ctx.root / r for r in PHOTO_ROOTS]
    have = next((r / "extra" for r in roots if (r / "extra" / "labels.csv").is_file()), None)
    if have is not None:
        return f"이미 있음, 받지 않음: {have} ({len(images_in(have))}장)"
    if not url.startswith("http"):
        raise Skip("EXTRA_DRIVE_URL 이 비어 있음(공개 사진 없이 진행)")
    target = next((r for r in reversed(roots) if _writable(r)), None)
    if target is None:
        raise RuntimeError(f"사진을 풀 수 있는 폴더가 없음: {[str(r) for r in roots]}")
    subprocess.run([ctx.py, "-m", "pip", "install", "-q", "gdown"], check=False)
    import gdown

    ctx.work.mkdir(parents=True, exist_ok=True)
    zip_path = ctx.work / "extra.zip"
    m = re.search(r"/d/([\w-]+)|[?&]id=([\w-]+)", url)  # fuzzy 없이 링크에서 파일 ID 를 꺼낸다
    file_id = (m.group(1) or m.group(2)) if m else url  # 링크 대신 ID 만 넣어도 된다
    gdown.download(id=file_id, output=str(zip_path), quiet=False)
    if not zip_path.is_file():
        raise RuntimeError("extra.zip 을 받지 못함. 링크가 '링크가 있는 모든 사용자' 공유인지 확인")
    got = zip_path.stat().st_size
    if size and got != size:
        raise RuntimeError(f"extra.zip 크기가 다름({got:,} != {size:,})")
    if sha and _sha256(zip_path) != sha:
        raise RuntimeError("extra.zip sha256 이 다름")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target)  # zip 안이 extra/... 구조
    zip_path.unlink()
    ctx._photo_dirs = None  # type: ignore[attr-defined]
    n = len(images_in(target / "extra"))
    note = "" if target == roots[-1] else " (/workspace/photos 에 쓸 수 없어 대체 폴더 사용)"
    return f"{got / 1e6:.0f}MB 받아 {target / 'extra'} 에 풂 ({n}장){note}"


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
            shutil.copyfile(p, target)
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
    """정답 CSV 를 찾는다. 사진 폴더의 labels.csv 와 extra/labels.csv 가 있으면 합쳐 results/labels_all.csv 로 쓴다."""
    import csv

    roots = [ctx.root] + [Path(r) if Path(r).is_absolute() else ctx.root / r for r in PHOTO_ROOTS]
    root = next((r for r in roots if (r / "labels.csv").is_file() or (r / "extra" / "labels.csv").is_file()), None)
    if root is None:
        return None
    files = [p for p in (root / "labels.csv", root / "extra" / "labels.csv") if p.is_file()]
    if len(files) == 1:
        return files[0]
    fields: list[str] = []
    rows: dict[str, dict] = {}  # 사진 이름 -> 줄. 두 파일에 같은 사진이 있으면 extra/labels.csv(뒤) 것을 쓴다
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields += [n for n in (reader.fieldnames or []) if n not in fields]
            for row in reader:
                rows[Path(row["file"].strip()).name] = row
    merged = ctx.results / "labels_all.csv"
    with merged.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows.values())
    return merged


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def cpu_info_of(ctx: Ctx):
    """실효 CPU 정보. 모델 패키지를 import 하지 않고 cpuinfo.py 만 파일로 불러온다(표준 라이브러리만 씀)."""
    import importlib.util

    path = ctx.root / "src" / "pair_vision" / "models" / "cpuinfo.py"
    spec = importlib.util.spec_from_file_location("pair_cpuinfo", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pair_cpuinfo"] = module  # dataclass 가 모듈을 찾을 수 있게
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.cpu_info()


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
    cpu = cpu_info_of(ctx)
    lines = [f"날짜: {datetime.now():%Y-%m-%d %H:%M}", f"플랫폼: {platform.platform()}", f"코어: {os.cpu_count()}",
             f"실효CPU: {cpu.effective}", f"affinity: {cpu.affinity if cpu.affinity is not None else '-'}",
             f"cgroup: {cpu.cgroup_raw or '제한 없음/읽지 못함'}",
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
    # tests/test_real_models.py 는 특정 책상 사진 8장(1.png~)을 가정하므로 임의의 사진을 넘기지 않는다(그 시험은 건너뜀).
    # 실제 모델 동작은 ⑤⑥⑧⑨ 단계가 확인한다.
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
    if ctx.quick and find_labels(ctx) is None:  # 정답이 있으면 ⑦ 정확도 평가를 위해 모든 사진을 돈다
        args += ["--limit", "4"]
    rc = sh(ctx, args, "05_compare.log", env=ctx.env(VISION_MODELS_DIR=str(ctx.models)), timeout=3600)
    if rc != 0:
        raise RuntimeError(f"compare 실패(코드 {rc}): {tail(ctx, '05_compare.log')}")
    text = (ctx.results / "logs" / "05_compare.log").read_text(encoding="utf-8")
    (ctx.results / "compare.md").write_text(text, encoding="utf-8")
    return "compare.md / compare.json"


# --- ⑥-0 진단: 스레드·배치·fp16 중 어느 설정이 빠른가 ---
DIAG_CONFIGS = (  # (이름, 스레드, 배치, fp16). 스레드: all=os.cpu_count(), auto=실효 CPU, 숫자=그 값
    ("스레드 모든 코어", "all", False, False),
    ("스레드 자동", "auto", False, False),
    ("스레드 4", "4", False, False),
    ("자동 + 배치", "auto", True, False),
    ("자동 + 배치 + fp16", "auto", True, True),
)
CPU_DIAG_CONFIGS = DIAG_CONFIGS[1:2] + DIAG_CONFIGS[3:4]  # CPU 는 느려서 둘만(fp16 은 의미 없음)
DIAG_SCALE_GPU = ("--examinees", "5", "--duration", "15", "--warmup", "4")
DIAG_SCALE_CPU = ("--examinees", "1", "--duration", "6", "--warmup", "2")
STAGE_NAMES = ("decode", "dfine", "owl", "hands", "face", "pose")


def _f(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def choose_setting(entries: list[dict]) -> dict:
    """진단 결과에서 가장 빠른 설정. fps(응시자당, 0.05 단위로 같은 값 취급)가 높고, 같으면 프레임당 단계 ms 합이 작은 쪽."""
    ok = [e for e in entries if e.get("fps") is not None]
    if not ok:
        raise RuntimeError("진단에서 측정에 성공한 설정이 없음")
    return min(ok, key=lambda e: (-round(e["fps"] * 20), e["total_ms"] if e.get("total_ms") is not None else float("inf")))


def _diag_entry(name: str, spec: str, batch: bool, fp16: bool, row: dict | None, note: str = "") -> dict:
    ms = {s: _f(row.get(f"ms_{s}")) for s in STAGE_NAMES} if row else {}
    total = round(sum(v for v in ms.values() if v is not None), 1) if any(v is not None for v in ms.values()) else None
    return {"name": name, "threads_spec": spec, "threads": (row or {}).get("threads", ""), "batch": batch, "fp16": fp16,
            "fps": _f(row.get("fps_min")) if row else None, "gpu_util_pct": _f((row or {}).get("gpu_util_pct")),
            "lat_p95_ms": _f((row or {}).get("lat_p95_ms")), "ms": ms, "total_ms": total, "note": note or (row or {}).get("note", "")}


def render_diag(entries: list[dict], chosen: dict, cpu_text: str) -> str:
    lines = ["# ⑥-0 진단: 스레드·배치·fp16", "", cpu_text, "",
             "workers 1, 웹캠 off, 응시자 5명(CPU 는 1명), 휴대폰 사진만. 응시자당 fps 가 같으면 단계 ms 합이 작은 쪽을 고름.", "",
             "| 설정 | 스레드 | 배치 | fp16 | fps | GPU % | p95 ms | " + " | ".join(f"{s} ms" for s in STAGE_NAMES) + " | ms 합 | 비고 |",
             "|" + "---|" * (7 + len(STAGE_NAMES) + 2)]
    for e in entries:
        cells = [e["name"], e["threads"] or "-", "on" if e["batch"] else "off", "on" if e["fp16"] else "off",
                 e["fps"], e["gpu_util_pct"], e["lat_p95_ms"], *(e["ms"].get(s) for s in STAGE_NAMES), e["total_ms"], e["note"]]
        lines.append("| " + " | ".join("-" if c is None or c == "" else str(c) for c in cells) + " |")
    lines += ["", f"**고른 설정: {chosen['name']}** (스레드 {chosen['threads'] or chosen['threads_spec']}, "
              f"배치 {'on' if chosen['batch'] else 'off'}, fp16 {'on' if chosen['fp16'] else 'off'}) -> ⑥ 용량이 이 값을 씀"]
    return "\n".join(lines) + "\n"


def step_diag(ctx: Ctx) -> str:
    require_ready(ctx)
    import csv

    dirs = photo_dirs(ctx)
    cpu = cpu_info_of(ctx)
    configs = DIAG_CONFIGS if ctx.gpu else CPU_DIAG_CONFIGS
    scale = list(DIAG_SCALE_GPU if ctx.gpu else DIAG_SCALE_CPU)
    out_dir = ctx.results / "diag"
    out_dir.mkdir(exist_ok=True)
    entries: list[dict] = []
    seen: dict[tuple, dict] = {}
    for i, (name, spec, batch, fp16) in enumerate(configs):
        actual = {"all": cpu.os_count, "auto": cpu.effective}.get(spec) or int(spec)
        key = (actual, batch, fp16)
        if key in seen:  # 실효 CPU 가 전체 코어와 같으면 '모든 코어'와 '자동'이 같은 설정
            entries.append({**seen[key], "name": name, "threads_spec": spec, "note": f"'{seen[key]['name']}' 와 같은 설정이라 다시 안 잼"})
            continue
        out = out_dir / f"diag_{i}"
        args = [ctx.py, "-u", "-m", "tools.bench.capacity", "--detector", ctx.detector, "--photos", str(dirs["phone"]),
                "--models-dir", str(ctx.models), "--out", str(out), "--workers", "1", "--webcam", "off",
                "--batch", "on" if batch else "off", "--fp16", "on" if fp16 else "off",
                "--worker-threads", str(0 if spec == "auto" else actual), *scale]
        rc = sh(ctx, args, f"06a_diag_{i}.log", timeout=1800)
        row = None
        if out.with_suffix(".csv").exists():
            rows = list(csv.DictReader(out.with_suffix(".csv").read_text(encoding="utf-8-sig").splitlines()))
            row = next((r for r in rows if r.get("ok") != "skipped" and r.get("fps_min") not in ("", None)), None)
        note = "" if row else f"측정 실패(코드 {rc}): {tail(ctx, f'06a_diag_{i}.log', 3)[:150]}"
        entry = _diag_entry(name, spec, batch, fp16, row, note)
        entries.append(entry)
        seen[key] = entry
    chosen = choose_setting(entries)
    ctx.diag = {"name": chosen["name"], "threads_spec": chosen["threads_spec"], "batch": chosen["batch"], "fp16": chosen["fp16"],
                "thread_cap": int(chosen["threads_spec"]) if chosen["threads_spec"].isdigit() else 0}
    (ctx.results / "diag.md").write_text(render_diag(entries, chosen, cpu.describe()), encoding="utf-8")
    info = {"os": cpu.os_count, "affinity": cpu.affinity, "cgroup": cpu.cgroup_raw, "effective": cpu.effective}
    (ctx.results / "diag.json").write_text(json.dumps({"cpu": info, "entries": entries, "chosen": ctx.diag}, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    return (f"{cpu.describe()}. 고른 설정: {chosen['name']} (배치 {'on' if chosen['batch'] else 'off'}, "
            f"fp16 {'on' if chosen['fp16'] else 'off'}, fps {chosen['fps']})")


# --- ⑥ 용량 측정 ---
def load_diag(ctx: Ctx) -> dict | None:
    if ctx.diag:
        return ctx.diag
    path = ctx.results / "diag.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))["chosen"]
        except (ValueError, KeyError):
            return None
    return None


def capacity_args(ctx: Ctx, out: Path, choice: dict | None) -> list[str]:
    """⑥ 용량의 capacity 인자. 진단이 고른 설정(스레드 상한·배치·fp16)을 쓰고, 없으면 기본값."""
    dirs = photo_dirs(ctx)
    choice = choice or {"batch": True, "fp16": ctx.gpu, "thread_cap": 0}
    args = [ctx.py, "-u", "-m", "tools.bench.capacity", "--detector", ctx.detector, "--photos", str(dirs["phone"]),
            "--models-dir", str(ctx.models), "--out", str(out), "--thread-cap", str(choice.get("thread_cap", 0))]
    if "webcam" in dirs:
        args += ["--webcam-photos", str(dirs["webcam"])]
    if not ctx.gpu:  # CPU 는 느려서 아주 작게(스크립트가 끝까지 도는지만 본다)
        return args + ["--workers", "1", "--batch", "on", "--fp16", "off", "--webcam", "server", "--examinees", "1",
                       "--duration", "6", "--warmup", "2"]
    batch, fp16 = ("on" if choice["batch"] else "off"), ("on" if choice["fp16"] else "off")
    if ctx.quick:  # 핵심 질문: 고른 설정에서 workers x 웹캠 x 인원. 미달하면 더 큰 인원은 건너뜀. 목표 10~15분
        return args + ["--workers", "1,2,4", "--batch", batch, "--fp16", fp16, "--webcam", "server,off",
                       "--examinees", "10,20,30,40", "--duration", "12", "--warmup", "4"]
    return args + ["--workers", "1,2,4,8", "--batch", "off,on", "--fp16", "off,on", "--webcam", "server,off",
                   "--examinees", "10,20,30,40"]


def step_capacity(ctx: Ctx) -> str:
    require_ready(ctx)
    out = ctx.results / f"capacity_{ctx.gpu_name}"
    choice = load_diag(ctx)
    rc = sh(ctx, capacity_args(ctx, out, choice), "06_capacity.log", timeout=4 * 3600)
    if rc != 0 or not out.with_suffix(".md").exists():
        raise RuntimeError(f"capacity 실패(코드 {rc}): {tail(ctx, '06_capacity.log')}")
    md = out.with_suffix(".md").read_text(encoding="utf-8").splitlines()
    failed = [ln for ln in md if "실패:" in ln]
    if failed:  # capacity.py 는 한 조합이 실패해도 끝까지 돌고 종료 코드 0 이므로 결과 표에서 확인한다
        raise RuntimeError(f"조합 실패 {len(failed)}건: {failed[0][:200]}. 로그: logs/06_capacity.log")
    rows: list[str] = []
    for i, ln in enumerate(md):
        if ln.startswith("| workers | 스레드/워커 | 배치 | fp16 | 웹캠 | 최대 인원"):  # 맨 위 요약 표
            for row in md[i + 2 :]:
                if not row.startswith("|"):
                    break
                cells = [c.strip() for c in row.strip("|").split("|")]
                rows.append(f"w{cells[0]}/스레드{cells[1]}/배치{cells[2]}/fp16 {cells[3]}/웹캠 {cells[4]}={cells[5]}")
            break
    used = f"진단이 고른 설정({choice['name']}) 사용" if choice else "진단 결과 없음: 기본 설정"
    return f"{out.name}.md/.csv. {used}. 최대 인원: {', '.join(rows[:8])}"


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
        raise Skip("labels.csv 없음(/workspace/photos/labels.csv, /workspace/photos/extra/labels.csv 또는 번들 --labels)")
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
def _scan_source(ctx: Ctx, dirs: dict[str, Path]) -> tuple[str, list[Path]]:
    """scan/ 폴더 > 공개 사진 extra/coco(최대 17장) > 합성 사진 순."""
    if "scan" in dirs:
        return "scan 폴더의 실제 사진", images_in(dirs["scan"])[:24]
    for r in PHOTO_ROOTS:
        base = Path(r) if Path(r).is_absolute() else ctx.root / r
        found = images_in(base / "extra" / "coco")
        if found:
            return "공개 사진(extra/coco)", found[:17]
    return "합성 사진(의미 없음)", images_in(synthetic_photos(ctx, 3))[:24]


def persons_agreement(labels: Path | None, files: list[Path], items: list[dict]) -> dict:
    """정답(persons 열)이 있는 사진에 대해 서버가 센 사람 수와 맞는 비율."""
    import csv

    if labels is None:
        return {"labeled": 0, "match": 0, "detail": []}
    with labels.open(newline="", encoding="utf-8-sig") as f:
        truth = {Path(r["file"].strip()).name: _f(r.get("persons")) for r in csv.DictReader(f)}
    detail = []
    for path, item in zip(files, items, strict=False):
        want = truth.get(path.name)
        if want is not None:
            detail.append({"file": path.name, "truth": int(want), "found": item["persons"]})
    return {"labeled": len(detail), "match": sum(1 for d in detail if d["truth"] == d["found"]), "detail": detail}


def step_scan(ctx: Ctx) -> str:
    require_ready(ctx)
    import httpx

    dirs = photo_dirs(ctx)
    source, files = _scan_source(ctx, dirs)
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
    empty = {"labeled": 0, "match": 0, "detail": []}
    agree = persons_agreement(find_labels(ctx), files, body.get("items", [])) if status == 200 else empty
    (ctx.results / "scan.json").write_text(json.dumps({"photos": len(files), "source": source, "calls_ms": [c[1] for c in calls],
                                                       "persons_agreement": agree, "response": body}, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    if status != 200 or calls[0][0] != 200:
        raise RuntimeError(f"/v1/scan 상태 {status}: {str(body)[:200]}")
    if bad.status_code != 400 or bad.json().get("code") != "BAD_SCAN":
        raise RuntimeError(f"잘못된 요청이 BAD_SCAN 이 아님: {bad.status_code} {bad.text[:100]}")
    reasons = [f"{r['code']}{r['names']}" for r in body["verdict"]["reasons"]]
    match = f", 사람 수 일치 {agree['match']}/{agree['labeled']}" if agree["labeled"] else ""
    return (f"사진 {len(files)}장({source}), verdict {body['verdict']['status']} {reasons}, 서버 {body['ms']}ms(왕복 {wall_ms}ms)"
            f"{match}, 잘못된 요청 400 BAD_SCAN 확인")


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
