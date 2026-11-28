"""GPU 합격 시험 노트북의 칸들(make_bundle.py 가 zip 의 크기·sha256 을 넣어 .ipynb 로 만든다)."""

from __future__ import annotations


def cells(bundle: dict) -> list[tuple[str, str]]:
    b = bundle
    return [
        ("markdown", f"""
# pAIr 추론 서버 GPU 합격 시험 (GPU 1장 · 응시자 20명 · 실시간)

**RunPod Pod 에서 이 노트북을 열고 맨 위 칸에 구글 드라이브 링크만 넣은 뒤 `Run > Run All Cells` 한 번이면 판정까지 끝납니다** (약 35~45분: 설치 5~15분, 예비 점검 3분, 본 시험 21분).

무엇을 하나: 압축 파일(서버 코드 + 모델) 받기·크기/sha256 검증 → 설치 → **예비 점검**(GPU·fp16·묶음 속도, 2명 1분 부하 - 설치 문제로 GPU 시간을 낭비하지 않게) → 서버·콜백 수신기 실행 → **응시자 20명 x 웹캠·휴대폰 x 초당 1장 x 20분** 부하 → 합격 기준 3개 자동 판정 → 서버 종료.

| 합격 기준(셋 다) | 한도 |
|---|---|
| 버린 사진 비율 (`pair_vision_frames_dropped_total`) | 1% 이하 |
| 경고 지연 p95 (`pair_vision_frame_latency_seconds`) | 3초 이하 |
| 처리된 사진 간격이 끊김 허용(휴대폰 3초·웹캠 2.5초)을 넘는 비율 (채널마다) | 1% 이하 |

- 묶음 번호: `{b['code_version']}` · 규칙 `{b['rules_version']}` · zip {b['bytes']:,} 바이트 · sha256 `{b['sha256'][:16]}…`
- Pod 조건: GPU 1장(측정 대상 GPU), 컨테이너 디스크 30GB 이상 권장, RunPod PyTorch 템플릿. 외부 포트 공개는 필요 없습니다(모두 Pod 안 127.0.0.1).
- 응시자 실사진은 쓰지 않습니다(합성 사진). 키·비밀은 실행할 때마다 새로 만들고 화면·파일에 남기지 않습니다.
- 칸이 실패하면 그 칸의 **빨간 메시지(원인·조치)** 와 마지막 출력을 보내 주세요. 끝나면 마지막 요약 덩어리를 복사해 보내고 **Pod 를 Stop/Terminate** 하세요(켜 두면 과금).
"""),
        ("markdown", "## 1. 설정 (여기만 고칩니다)"),
        ("code", f"""
# 구글 드라이브 공유 링크('링크가 있는 모든 사용자'). 압축 파일을 JupyterLab 에 직접 올렸다면 비워 둬도 됩니다.
DRIVE_URL = ""

STUDENTS = 20          # 응시자 수(각 응시자 = 웹캠 1 + 휴대폰 1, 초당 1장씩)
DURATION_S = 1200      # 판정 구간(초). 합격 기준은 20분 이상
WARMUP_S = 60          # 판정에서 빼는 앞 구간(모델 첫 실행·웹캠 보정)

# --- 아래는 고치지 않습니다(압축 파일과 짝) ---
BUNDLE_NAME = "pair-vision-gpu-test.zip"
BUNDLE_BYTES = {b['bytes']}
BUNDLE_SHA256 = "{b['sha256']}"
CODE_VERSION = "{b['code_version']}"
"""),
        ("code", r'''
import hashlib, json, os, secrets, shutil, signal, subprocess, sys, time, urllib.request as ur, zipfile, atexit
from pathlib import Path

# 이 PC 의 CPU 흐름 확인용 덮어쓰기(Pod 에서는 쓰지 않음)
DRIVE_URL = os.environ.get("PAIR_TEST_DRIVE_URL", DRIVE_URL)
STUDENTS = int(os.environ.get("PAIR_TEST_STUDENTS", STUDENTS))
DURATION_S = float(os.environ.get("PAIR_TEST_DURATION", DURATION_S))
WARMUP_S = float(os.environ.get("PAIR_TEST_WARMUP", WARMUP_S))
ALLOW_CPU = os.environ.get("PAIR_TEST_ALLOW_CPU") == "1"   # GPU 없이 흐름만 확인(판정 수치는 의미 없음)
PRECHECK_STUDENTS, PRECHECK_S = 2, float(os.environ.get("PAIR_TEST_PRECHECK", 60))


class 시험중단(RuntimeError):
    pass


def fail(msg):
    """원인과 조치를 한국어로 알리고 Run All 을 멈춘다."""
    print("\n" + "!" * 70 + "\n[중단] " + msg + "\n" + "!" * 70, file=sys.stderr)
    raise 시험중단(msg)


def writable(p):
    try:
        p.mkdir(parents=True, exist_ok=True)
        (p / ".w").write_text("x"); (p / ".w").unlink()
        return True
    except OSError:
        return False


WORK = next((Path(c) for c in (os.environ.get("PAIR_TEST_WORK"), str(Path.home() / "pair-vision-gpu-test"), "/tmp/pair-vision-gpu-test")
             if c and writable(Path(c))), None) or fail("쓸 수 있는 작업 폴더가 없습니다(홈·/tmp 권한 확인)")
ROOT = WORK / "pAIr-vision"
VENV_PY = str(WORK / "venv-vision" / "bin" / "python")
LOG = WORK / "vision-server.log"
SINK_LOG = WORK / "sink.log"
IMAGES = WORK / "images"
REPORT = WORK / "gpu-test-report.json"
PROCS = []


def stop_all():
    for p in PROCS:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except (OSError, AttributeError):
                p.terminate()
    for p in PROCS:
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
    PROCS.clear()


atexit.register(stop_all)
smi = shutil.which("nvidia-smi")
GPU_NAME = subprocess.run([smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True,
                          text=True).stdout.strip() if smi else ""
CPUS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 2)
free_gb = shutil.disk_usage(WORK).free / 1e9
print(f"작업 폴더 {WORK} | 남은 디스크 {free_gb:.0f}GB | CPU {CPUS}코어 | python {sys.version.split()[0]}")
print("GPU:", GPU_NAME or "없음")
if not GPU_NAME and not ALLOW_CPU:
    fail("GPU 가 보이지 않습니다(nvidia-smi 없음). Pod 를 GPU 로 만들었는지 확인하세요. GPU 없이 시험하면 판정이 의미 없습니다.")
if free_gb < 15:
    fail(f"디스크가 부족합니다({free_gb:.0f}GB). 압축 파일·패키지에 약 15GB 가 필요합니다 - Pod 의 컨테이너 디스크를 늘리세요.")
'''),
        ("markdown", "## 2. 압축 파일 받기 · 크기/sha256 검증 · 풀기"),
        ("code", r'''
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_bundle(path):
    size = path.stat().st_size
    if size != BUNDLE_BYTES:
        return f"크기가 다릅니다: {size:,} (기대 {BUNDLE_BYTES:,})"
    digest = sha256_of(path)
    if digest != BUNDLE_SHA256:
        return f"sha256 이 다릅니다: {digest} (기대 {BUNDLE_SHA256})"
    return None


def download(dest):
    if not DRIVE_URL:
        fail("DRIVE_URL 이 비어 있고 압축 파일도 없습니다. 1번 칸에 구글 드라이브 공유 링크를 넣거나 "
             f"JupyterLab 왼쪽 파일 창으로 {BUNDLE_NAME} 을 올린 뒤 다시 실행하세요.")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "gdown"], check=True)
    import importlib
    import re
    import gdown
    importlib.reload(gdown)
    # 공유 링크 모양(…/file/d/<ID>/view, open?id=<ID>, uc?id=<ID>)을 모두 uc?id=<ID> 로 바꾼다(옛 gdown 은 fuzzy 가 없다)
    m = re.search(r"/d/([A-Za-z0-9_-]{10,})", DRIVE_URL) or re.search(r"[?&]id=([A-Za-z0-9_-]{10,})", DRIVE_URL)
    url = f"https://drive.google.com/uc?id={m.group(1)}" if m and "drive.google" in DRIVE_URL else DRIVE_URL
    print("받는 중(1GB 안팎, 몇 분)...", flush=True)
    try:
        try:
            gdown.download(url, str(dest), quiet=False, fuzzy=True)
        except TypeError:
            gdown.download(url, str(dest), quiet=False)
    except Exception as exc:
        fail(f"내려받기 실패: {exc!r}\n링크가 '링크가 있는 모든 사용자' 공유인지, 하루 내려받기 한도에 걸리지 않았는지 확인하세요."
             " 안 되면 PC 에서 받아 JupyterLab 으로 직접 올리세요.")
    if not dest.is_file() or dest.stat().st_size < 1_000_000:
        fail("받은 파일이 없거나 너무 작습니다(구글 드라이브 경고 페이지를 받았을 수 있음). 공유 설정을 확인하세요.")


MARKER = ROOT / ".bundle-sha256"
if MARKER.is_file() and MARKER.read_text().strip() == BUNDLE_SHA256 and (ROOT / "src" / "pair_vision").is_dir():
    print("이미 풀려 있음(같은 압축 파일) -> 건너뜁니다:", ROOT)
else:
    cands = [Path(os.environ.get("PAIR_TEST_BUNDLE", "")), Path.cwd() / BUNDLE_NAME, WORK / BUNDLE_NAME,
             Path.home() / BUNDLE_NAME, Path("/workspace") / BUNDLE_NAME]
    zpath = next((p for p in cands if str(p) not in ("", ".") and p.is_file()), None)
    if zpath is not None and check_bundle(zpath):
        print("있던 파일이 이 노트북과 짝이 아닙니다 ->", check_bundle(zpath), "-> 새로 받습니다")
        zpath = None
    if zpath is None:
        zpath = WORK / BUNDLE_NAME
        download(zpath)
        problem = check_bundle(zpath)
        if problem:
            fail(problem + "\n링크의 압축 파일과 이 노트북이 서로 다른 판입니다. 둘을 같은 때 받은 것으로 맞추세요.")
    print("검증 통과:", zpath, f"{BUNDLE_BYTES:,} 바이트")
    if ROOT.exists():
        shutil.rmtree(ROOT)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(ROOT)
    MARKER.write_text(BUNDLE_SHA256)
info = json.loads((ROOT / "bundle-info.json").read_text())
print(f"코드 {info['code_version']} | 규칙 {info['rules_version']} | 파일 {len(info['files'])}개 (만든 때 {info['made_at']})")
bad = [rel for rel, digest in info["files"].items() if sha256_of(ROOT / rel) != digest]
if bad:
    fail(f"풀린 파일 {len(bad)}개가 묶을 때와 다릅니다(예: {bad[:3]}). 디스크·압축 해제 문제 - 이 칸을 다시 실행하세요.")
print("풀린 파일 sha256 모두 일치")
os.chdir(ROOT)
'''),
        ("markdown", "## 3. 설치 (시스템 라이브러리 · 파이썬 패키지 · 모델 sha256 검증)\n`tools/runpod/bootstrap.sh` 를 그대로 씁니다(GPU 가 있으면 onnxruntime-gpu 로 교체, torch 가 2.6 미만이면 가상환경에 2.6). 처음엔 5~15분."),
        ("code", r'''
t0 = time.time()
boot = subprocess.Popen(["bash", "tools/runpod/bootstrap.sh"], cwd=ROOT, env={**os.environ, "WORK": str(WORK)},
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
tail = []
for line in boot.stdout:
    print(line, end="")
    tail = (tail + [line])[-30:]
if boot.wait() != 0:
    fail("설치 실패. 위 출력의 마지막 오류를 보내 주세요(인터넷·디스크·pip 문제가 흔합니다).")
print(f"설치 끝 ({time.time() - t0:.0f}초)")
'''),
        ("markdown", "## 4. 예비 점검 (약 2~3분) - GPU 확인, fp16 결과 비교, 묶음 속도 곡선\n결과가 fp32 와 다른 모델은 자동으로 fp32 로 둡니다(`VISION_FP16_SKIP`, 보고서에 기록)."),
        ("code", r'''
subprocess.run([VENV_PY, "tools/gpu_test/make_images.py", "--out", str(IMAGES)], cwd=ROOT, check=True)
pre = subprocess.run([VENV_PY, "tools/gpu_test/preflight.py", "--models-dir", str(ROOT / "models"), "--images", str(IMAGES),
                      "--out", str(WORK / "preflight.json")] + (["--quick"] if ALLOW_CPU and not GPU_NAME else []),
                     cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True)
print("\n".join(l for l in pre.stdout.splitlines() if "Loading weights" not in l))
if pre.returncode not in (0, 1):
    print(pre.stderr[-3000:])
    fail("예비 점검(모델 직접 실행) 실패 - 위 오류를 보내 주세요(CUDA·torch·onnxruntime 설치 문제일 가능성).")
PREFLIGHT = json.loads((WORK / "preflight.json").read_text())
FP16_SKIP = ",".join(PREFLIGHT.get("fp16", {}).get("fp16_skip", []))
env_ok = PREFLIGHT["env"]
if not ALLOW_CPU:
    if not env_ok["cuda"]:
        fail("torch 가 GPU(CUDA)를 못 잡습니다. bootstrap 출력의 'CUDA 사용 가능' 줄과 torch 버전을 확인하세요.")
    if not PREFLIGHT.get("rtmw_gpu"):
        fail("ONNX Runtime 이 GPU 를 못 씁니다(RTMW 가 CPU 로 돎). providers: " + str(env_ok["ort_providers"]) +
             " - onnxruntime-gpu 와 CUDA/cuDNN 버전 확인이 필요합니다.")
    if not PREFLIGHT.get("device_preprocess"):
        fail("전처리가 GPU 로 돌지 않습니다(device_preprocess=false). configs/rules.default.yaml 의 models.fast_preprocess 를 확인하세요.")
    if not PREFLIGHT.get("gpu_jpeg_decode"):
        print("주의: torchvision nvJPEG 을 못 써 JPEG 만 CPU(PIL)로 풀고 GPU 로 올립니다(나머지 전처리는 GPU). 결과 보고에 적습니다.")
print("예비 점검 통과 | fp32 로 둘 모델:", FP16_SKIP or "없음", "| 전처리 GPU:", PREFLIGHT.get("device_preprocess"),
      "| JPEG GPU 풀기:", PREFLIGHT.get("gpu_jpeg_decode"))
'''),
        ("markdown", "## 5. 서버 · 콜백 수신기 실행\n키·서명 비밀은 이번 실행에만 쓰는 임의 값입니다(출력하지 않음). 서버는 시작할 때 모델 sha256 을 `configs/models.lock` 과 대조합니다."),
        ("code", r'''
stop_all()
API_KEY, CB_SECRET = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
base_env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
sink = subprocess.Popen([VENV_PY, "tools/replay/sink.py", "--port", "9000"], cwd=ROOT,
                        env={**base_env, "VISION_CALLBACK_SECRET": CB_SECRET}, stdout=open(SINK_LOG, "w"),
                        stderr=subprocess.STDOUT, start_new_session=True)
PROCS.append(sink)
SERVER_ENV = {
    "VISION_API_KEY": API_KEY, "VISION_DETECTOR": "real", "VISION_MODELS_DIR": str(ROOT / "models"), "VISION_VERIFY_MODELS": "1",
    "VISION_BATCH": "1", "VISION_WORKERS": "1", "VISION_MAX_SESSIONS": "100", "VISION_PREP_THREADS": str(max(2, min(8, CPUS))),
    "VISION_FP16_SKIP": FP16_SKIP, "VISION_HOST": "127.0.0.1", "VISION_PORT": "8000",
    "VISION_CALLBACK_URL": "http://127.0.0.1:9000/internal/vision/events", "VISION_CALLBACK_SECRET": CB_SECRET,
}
server = subprocess.Popen([VENV_PY, "-m", "pair_vision"], cwd=ROOT, env={**base_env, **SERVER_ENV}, stdout=open(LOG, "w"),
                          stderr=subprocess.STDOUT, start_new_session=True)
PROCS.append(server)


def readyz():
    req = ur.Request("http://127.0.0.1:8000/readyz", headers={"Authorization": f"Bearer {API_KEY}"})
    return json.load(ur.urlopen(req, timeout=5))


READY = None
t0 = time.time()
while time.time() - t0 < 900:
    time.sleep(3)
    if server.poll() is not None:
        break
    try:
        READY = readyz()
        if READY.get("modelsLoaded"):
            break
    except Exception:
        continue
if not READY or not READY.get("modelsLoaded"):
    print("".join(LOG.read_text(errors="replace").splitlines(keepends=True)[-25:]))
    fail("서버가 준비되지 않았습니다. 위 로그에 '모델 검증 실패'가 있으면 모델 파일이 lock 과 다른 것이고, 그 밖의 오류는 그대로 보내 주세요.")
print(f"서버 준비({time.time() - t0:.0f}초):", {k: READY[k] for k in ("modelsLoaded", "gpu", "rulesVersion", "maxSessions")})
loaded = [l for l in LOG.read_text(errors="replace").splitlines()
          if any(w in l for w in ("모델 적재 완료", "묶음 추론 엔진", "예열", "전처리 전부"))]
print("\n".join(loaded))
if not READY["gpu"] and not ALLOW_CPU:
    fail("/readyz 가 gpu:false 입니다(D-FINE·OWLv2 또는 RTMW 가 CPU 로 돎). 위 '모델 적재 완료' 줄과 4번 예비 점검 결과를 보내 주세요.")
'''),
        ("markdown", "## 6. 짧은 부하 확인 (2명 x 1분) - 본 시험 전에 콜백·서명·지표가 정상인지"),
        ("code", r'''
def load_test(students, duration, warmup, out, progress=60, require_events=False):
    cmd = [VENV_PY, "tools/gpu_test/load_test.py", "--url", "http://127.0.0.1:8000", "--api-key", API_KEY,
           "--students", str(students), "--duration", str(duration), "--warmup", str(warmup), "--images", str(IMAGES),
           "--sink-url", "http://127.0.0.1:9000", "--out", str(out), "--progress-every", str(progress)]
    if ALLOW_CPU:
        cmd.append("--allow-cpu")
    if require_events:
        cmd.append("--require-events")
    p = subprocess.Popen(cmd, cwd=ROOT, env=base_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    code = p.wait()
    if code == 3 or not Path(out).is_file():
        print("".join(LOG.read_text(errors="replace").splitlines(keepends=True)[-15:]))
        fail("부하 시험 도구가 실패했습니다(위 출력·서버 로그 확인).")
    if server.poll() is not None:
        fail("부하 중에 서버가 죽었습니다. 서버 로그 마지막: " + LOG.read_text(errors="replace")[-1500:])
    return json.loads(Path(out).read_text())


smoke = load_test(PRECHECK_STUDENTS, PRECHECK_S, 15, WORK / "smoke.json", progress=30, require_events=True)
if not smoke["valid"]:
    fail("짧은 부하에서 정합성 실패: " + json.dumps(smoke["sanity"], ensure_ascii=False) +
         " - 콜백(수신기 도착 0)·모델 오류가 있으면 본 시험을 해도 판정이 무효입니다. 위 표와 서버 로그를 보내 주세요.")
if not ALLOW_CPU and smoke["counts"]["processed"] < 0.5 * smoke["counts"]["received"]:
    fail("2명에서도 절반 넘게 처리하지 못했습니다 - 20명 시험은 의미가 없습니다(GPU 가 아닌 CPU 로 도는지 확인).")
print("\n짧은 부하 확인 통과 -> 본 시험을 시작합니다")
'''),
        ("markdown", "## 7. 본 시험: 응시자 20명 x 2채널 x 초당 1장 x 20분 (+ 앞 1분 제외)\n1분마다 진행 상황이 찍힙니다. 끝나면 판정 표가 나옵니다."),
        ("code", r'''
t0 = time.time()
RESULT = load_test(STUDENTS, DURATION_S, WARMUP_S, REPORT)
print(f"\n본 시험 끝 ({(time.time() - t0) / 60:.1f}분)")
'''),
        ("markdown", "## 8. 결과 요약 (이 덩어리를 복사해 보내 주세요)"),
        ("code", r'''
r, c = RESULT, RESULT["criteria"]
pct = lambda v: "-" if v is None else f"{v:.2%}"
lines = [
    "=" * 20 + f" pAIr GPU 합격 시험: {r['verdict']} " + "=" * 20,
    f"코드 {info['code_version']} | 규칙 {info['rules_version']} | GPU {GPU_NAME or '없음'} | CPU {CPUS}코어 | 정밀도 {PREFLIGHT.get('precision')} | fp32 로 둔 모델 {FP16_SKIP or '없음'} | 전처리 GPU {PREFLIGHT.get('device_preprocess')} (JPEG GPU 풀기 {PREFLIGHT.get('gpu_jpeg_decode')})",
    f"응시자 {r['students']}명 x 2채널 x 초당 {r['fps']}장, 판정 구간 {r['measured_s']}초",
    f"[기준1] 버린 사진 {pct(c['drop']['value'])} (한도 1%) -> {'통과' if c['drop']['ok'] else '실패'}",
    f"[기준2] 지연 p95 {c['latency_p95']['value'] and round(c['latency_p95']['value'], 2)}초, 3초 안 {pct(c['latency_p95']['within_3s_ratio'])} (한도 p95 3초) -> {'통과' if c['latency_p95']['ok'] else '실패'}",
]
for ch, g in c["gap_over"]["by_channel"].items():
    lines.append(f"[기준3] 끊김 초과 {ch}(>{g['limit_s']}초) {pct(g['over_ratio'])}, 간격 p95 {g['p95_s'] and round(g['p95_s'], 2)}초 (한도 1%) -> "
                 f"{'통과' if g['over_ratio'] <= 0.01 and g['count'] else '실패'}")
n = r["counts"]
lines += [
    "정합성: " + ", ".join(f"{k}={'OK' if v else '실패'}" for k, v in r["sanity"].items()) + " | 모델 sha256: 서버 시작 때 lock 대조 통과",
    f"받음 {n['received']} 처리 {n['processed']} 버림 {n['dropped']} | 이벤트 {n['events']} 콜백 실패 {n['callback_failures']} 수신기 도착(서명 검증) {n['sink_received']} | 모델 오류 {n['detector_errors']}",
    f"GPU 사용률 평균 {r['gpu_util']['avg']}% 최대 {r['gpu_util']['max']}% | GPU 메모리 최대 {r['gpu_util']['mem_max_mib']}MiB | load1 평균 {r['load1_avg']}",
    "단계(프레임당 ms / 점유 초/초 / 평균 묶음): " + "; ".join(
        f"{k} {v['ms_per_frame']}ms/{v['busy']}/{v['avg_batch']}" for k, v in sorted(r["stages"].items(), key=lambda kv: -kv[1]["busy"])),
    "묶음 속도 곡선(장당 ms): " + json.dumps(PREFLIGHT.get("batch_curve_ms_per_image"), ensure_ascii=False),
    "fp16 비교: " + json.dumps(PREFLIGHT.get("fp16"), ensure_ascii=False),
]
if r["verdict"] == "합격":
    lines.append("=> 합격: GPU 1장으로 20명 실시간 경고 유지 가능(이 조건 기준).")
elif r["verdict"] == "불합격":
    lines.append("=> 불합격: 의도(docs/intent/realtime-vision-20.md)에 따라 회차 뒤 사후 분석으로 방향 전환 검토.")
else:
    lines.append("=> 판정 무효: 정합성 항목이 실패했습니다. 수치를 판정에 쓰지 말고 원인을 먼저 확인합니다.")
lines.append("=" * 70)
summary = "\n".join(lines)
print(summary)
(WORK / "gpu-test-summary.txt").write_text(summary)
print("\n전체 보고서:", REPORT, "| 예비 점검:", WORK / "preflight.json", "| 서버 로그:", LOG)
'''),
        ("markdown", "## 9. 서버 종료\n끈 뒤 **RunPod 화면에서 Pod 를 Stop/Terminate** 하세요."),
        ("code", r'''
stop_all()
print("서버·수신기 종료. 보고서는", WORK, "에 있습니다(Pod 를 지우면 사라지니 필요하면 내려받으세요).")
'''),
    ]
