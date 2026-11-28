# pAIr-vision

pAIr 온라인 시험 서비스의 **서버 추론 감독용 추론 서버**(RunPod GPU 에서 돌릴 예정). 백엔드(`pAIr-back`)가 보낸 웹캠·휴대폰 사진을 최신 한 장씩 분석해, 규칙에 걸린 이상 상태를 서명된 HTTP 콜백으로 백엔드에 알린다.

- 계약: pAIr-docs `설계/서버_추론_감독_설계.md` §3·§6·§9 (사본 `contract.md`). **계약을 바꾸려면 문서 담당 세션을 통해 고친다.**
- 이 저장소의 현재 범위: 계약 뼈대 + 가짜 모델 + **실제 모델 연결(마일스톤 A: 휴대폰 채널, B: 웹캠 채널)**. 기본은 가짜 모델(`VISION_DETECTOR=fake`)이고, `VISION_DETECTOR=real` 이면 D-FINE-s + OWLv2 + MediaPipe Hands(2026-11-26 확정, 그림자 모드 후 재확정)를 쓴다.
- 추론 서버는 S3·DB 에 접근하지 않고, 사진을 디스크에 쓰지 않으며, 사진·키·서명을 로그에 남기지 않는다(접근 로그도 꺼 둠: 경로에 `visionKey` 가 있다).

## 구현된 것

| 구분 | 내용 |
|---|---|
| `WS /v1/stream` | 바이너리 프레임(version 1, channel, seq, recvMs, visionKey 16B, JPEG). 응시자·채널마다 큐 길이 1, 같은 seq 버림, 모르는 키는 `UNKNOWN_SESSION` 을 키마다 10초에 1번. 텍스트: `state`(exam 세션마다 2초), `check`, `error` |
| REST | `POST /v1/sessions/{key}/start`(멱등·`recalibrate`·`OVER_CAPACITY`+`Retry-After`·`DRAINING`), `/mode`, `/end`, `GET /healthz` `/readyz` `/metrics`, `POST /v1/admin/drain` |
| 인증 | `Authorization: Bearer`, `VISION_API_KEY` 와 `VISION_API_KEY_NEXT` 동시 유효 |
| 이벤트 콜백 | `POST {VISION_CALLBACK_URL}`, HMAC-SHA256 서명, 결정적 `clientId`, 재시도 1·2·4·8·16·30초→30초 간격 10분, 대기 큐 1000건, 400/404/409(+413/422) 는 재시도 안 함, final 로 지속시간 갱신 |
| 규칙 엔진 | `HANDS_OUT` 7초 / `HAND_OUT_ONE` 10초 / 끊김 허용 3초 / 쿨다운 10초 / 전환 시 합산 없음 / 웹캠 보정 3초. 값은 `configs/rules.default.yaml`, 이벤트에 `rulesVersion` 기록 |
| 가짜 모델 | `FakeDetector`: 환경 변수 시나리오 또는 프레임 안 `FAKE:<STATE>` 표식으로 손·키보드 위치를 내놓음 |

| 실제 모델 | `RealDetector`: D-FINE-s(키보드·마우스·노트=COCO `book`, 프레임마다) · OWLv2(태블릿·스마트워치·다른 휴대폰, 세션마다 `owl_interval_ms` 주기) · MediaPipe Hands(최대 4개 검출 → 가까운 중복 합침 → 점수 상위 2개만 손 규칙에 사용) |
| 물체 규칙 | `TABLET`·`NOTE`·`SMARTWATCH` 는 **`source=PHONE_CAM` 으로만** 만든다. 손 3개 이상은 `MANY_HANDS` 로 **그림자 모드**(콜백 없음, `pair_vision_shadow_events_total` 지표와 로그만 - 계약에 없는 유형이라) |

| 웹캠 규칙 | 프론트 `pcRules.ts` 포팅: `NO_FACE` `MULTI_FACE` `HEAD_TURN` `HEAD_DOWN` `TORSO_SHIFT` `TORSO_TURN` (`source=PC_CAM`). **`GAZE_AWAY`(시선)는 2026-11-26 사용자 결정으로 기본 꺼짐**(`webcam.rules.gaze_away.enabled`, 아래 시선 측정 참고). MediaPipe 얼굴 랜드마커(최대 3얼굴·블렌드셰이프·변환 행렬)+포즈 랜드마커 lite 로 특징값을 만들고, 보정(3초) 중앙값 대비 차이로 판정한다. 임계값은 프론트 `PC_THRESHOLDS` 와 같고 `rules.default.yaml` 의 `webcam:` 에서 바꾼다 |

아직 없는 것: `PHONE_MOVED`, "다른 휴대폰" 이벤트(계약에 유형이 없어 검출만 하고 기록하지 않는다).

## 실행

```powershell
uv sync
uv run pytest
$env:VISION_API_KEY = "dev-key"; uv run python -m pair_vision     # http://localhost:8000
```

Docker(CPU, 가짜 모델):

```powershell
docker compose up --build
curl.exe -H "Authorization: Bearer dev-key" http://localhost:8000/readyz
```

실제 모델 이미지(가중치는 이미지에 넣지 않고 `models/` 를 `/models` 로 읽기 전용 마운트; `uv.lock` 은 `real` 선택 의존성까지 갱신해 두었다):

```powershell
docker compose --profile real up --build vision-real     # CPU 로 확인. GPU 는 compose 의 deploy 주석을 풀고 NVIDIA 런타임이 있는 호스트에서
docker build --build-arg UV_EXTRAS="--extra real" --build-arg BASE_IMAGE=<CUDA 런타임+Python 3.12 이미지> -t pair-vision:gpu .   # RunPod 용
```

> 실제 모델 이미지(`UV_EXTRAS="--extra real"`, 약 19.6GB)는 이 PC(Docker Desktop, CPU)에서 빌드해 `/readyz` `modelsLoaded:true` 와 사진 재생(`TABLET`·`SMARTWATCH` 이벤트, 분석 실패 0)까지 확인했다. 빌드로 찾아 고친 것: 시스템 라이브러리(`libegl1` `libgles2` `libxcb1` 등)와 `scipy`(OWLv2 전처리). **GPU(CUDA 베이스 이미지)로는 아직 빌드·실행하지 못했다** - RunPod 에서 `gpu:true` 를 확인해야 한다. Git Bash 에서 `docker run -e VISION_MODELS_DIR=/models` 를 쓸 때는 `MSYS_NO_PATHCONV=1` 이 필요하다(경로가 `C:/Program Files/Git/models` 로 바뀐다).

콜백 수신기와 부하 재생기까지 보려면:

```powershell
docker compose --profile demo up -d --build
uv run python tools/replay/replay.py --api-key dev-key --streams 3 --scenario "NORMAL:1,BOTH_OUT:12" --duration 12
docker compose logs sink      # 서명이 검증된 HANDS_OUT 이벤트
```

### 실제 모델로 실행 (이 PC, CPU)

가중치는 git 에 없다. `models/` 폴더(또는 `VISION_MODELS_DIR`)에 `dfine-small-coco/`, `owlv2-base-patch16-ensemble/`, `hand_landmarker.task`, `face_landmarker.task`, `pose_landmarker_lite.task` 가 있어야 하고(pip `mediapipe` 에는 모델 파일이 들어 있지 않다. 얼굴·포즈는 프론트 `visionAssets.ts` 와 같은 Google 저장소 주소에서 받는다)(출처·sha256·라이선스는 `configs/models.lock`), 선택 의존성이 필요하다.

```powershell
uv pip install --python .venv-bench\Scripts\python.exe fastapi "uvicorn[standard]" websockets httpx prometheus-client pytest-asyncio   # 서버 의존성(.venv-bench 에 torch·transformers·mediapipe 가 이미 있을 때)
# 새 환경이면: uv sync --extra real   (uv.lock 은 `uv lock` 으로 다시 만들어야 한다)
$env:PYTHONPATH = "src"; $env:VISION_CALLBACK_SECRET = "dev-callback-secret"
Start-Process .\.venv-bench\Scripts\python.exe -ArgumentList "tools/replay/sink.py --port 9000"
$env:VISION_API_KEY = "dev-key"; $env:VISION_DETECTOR = "real"; $env:VISION_CALLBACK_URL = "http://localhost:9000/internal/vision/events"
$env:VISION_RULES_PATH = "configs/rules.local-cpu.yaml"    # CPU 전용(아래 참고)
.\.venv-bench\Scripts\python.exe -m pair_vision                # 모델 적재(sha256 검증 포함) 약 10~35초
.\.venv-bench\Scripts\python.exe tools/replay/replay.py --api-key dev-key --photos C:/dev/bench-photos --photo-names 7.png,5.png --hold 14 --duration 30 --settle 6
```

- `/readyz` 의 `modelsLoaded` 가 true 가 되면 준비된 것이다. 가중치가 없거나 sha256 이 다르면 서버가 시작하지 않는다(`VISION_VERIFY_MODELS=0` 으로 검증만 끌 수 있다).
- **CPU 주의**: 이 PC 에서 OWLv2 한 번이 약 3초라 `rules.default.yaml`(OWLv2 5초 주기)로 돌리면 프레임 간격이 손 규칙의 끊김 허용(3초)을 넘어 손 이벤트가 안 난다. 로컬 CPU 시험은 OWLv2 주기를 30초로 늘린 `configs/rules.local-cpu.yaml` 을 쓴다(운영 값 아님, GPU 에서는 약 0.15초라 해당 없음).
- 통합 시험: `$env:VISION_TEST_PHOTOS = "C:/dev/bench-photos"; .\.venv-bench\Scripts\python.exe -m pytest tests/test_real_models.py`. 가중치·사진이 없으면 자동으로 건너뛴다.
- 휴대폰 프레임에 불필요한 여백이 있으면 손 검출이 떨어진다(측정에서 확인). 거치대·해상도·구도를 정할 때 반영한다.

60스트림 부하는 `VISION_MAX_SESSIONS=60` 으로 띄우고 `--streams 60`.

## 환경 변수

| 이름 | 기본 | 설명 |
|---|---|---|
| `VISION_API_KEY` / `VISION_API_KEY_NEXT` | (필수 / 없음) | 백엔드 Bearer 키, 무중단 교체용 두 개 |
| `VISION_CALLBACK_URL` | 없음 | 백엔드 `/internal/vision/events` 전체 주소. 없으면 이벤트를 보내지 않고 버린 수만 센다 |
| `VISION_CALLBACK_KEY_ID` / `VISION_CALLBACK_SECRET` | `k1` / (URL 이 있으면 필수) | 서명 키 번호와 비밀 |
| `VISION_MAX_SESSIONS` | 40 | 동시 세션 상한 (**실측 M4·M5 전 임시값**) |
| `VISION_RULES_PATH` | `configs/rules.default.yaml` | 규칙 파일 |
| `VISION_DETECTOR` / `VISION_FAKE_SCENARIO` | `fake` / `normal` | `fake` 또는 `real`. 가짜 시나리오(`hands_out` `one_hand_out` `flicker` 또는 `NORMAL:3,BOTH_OUT:10,NORMAL:1`; 프레임 표식 `FAKE:TABLET`·`FAKE:NOTE` 도 있다) |
| `VISION_MODELS_DIR` / `VISION_MODELS_LOCK` | `./models` / `configs/models.lock` | 모델 폴더(git 제외)와 검증용 lock |
| `VISION_VERIFY_MODELS` | `1` | 시작할 때 모델 파일 sha256 을 lock 과 대조(`0` 이면 건너뜀) |
| `VISION_WORKERS` / `VISION_WORKER_THREADS` | `1` / `0` | 모델 프로세스(워커) 수와 워커당 CPU 코어·스레드 수(0=사용 가능 코어/워커 수). 2 이상이면 응시자별로 워커를 고정 배정해 병렬 처리한다. **실측(RTX A5000 Pod): 워커 1개일 때 초당 약 5장에서 막혀 응시자 2명까지만 처리됐다(GPU 3~17%, CPU load 높음).** 워커 병렬화 효과는 Pod 에서 다시 재야 한다 |
| `VISION_BATCH` | `0` (`tools/runpod/start.sh` 는 `1`) | `1` 이면 **묶음 추론**: 모델을 한 프로세스에 두고 응시자 사이 프레임을 단계별로 묶어 돌린다(아래 절). `VISION_WORKERS` 는 무시 |
| `VISION_BATCH_MAX` / `VISION_BATCH_WAIT_MS` / `VISION_OWL_MAX_BATCH` | `16` / `15` / `4` | 단계당 최대 묶음, GPU 가 놀 때 첫 프레임 뒤 더 모으는 시간, OWLv2(960px) 묶음 상한 |
| `VISION_PREP_THREADS` | `4` | JPEG 풀기·CPU 전처리 스레드 |
| `VISION_DEVICE_PREPROCESS` | 비움(자동) | 전부 GPU 전처리(JPEG nvJPEG 풀기·RTMW 자르기·YOLOX 레터박스). 비우면 GPU + `models.fast_preprocess` 일 때 켜짐, `1`/`0` 으로 강제 |
| `VISION_FP16_SKIP` | 없음 | `models.fp16: true` 여도 fp32 로 둘 모델(`dfine`, `owlv2`). GPU 점검에서 결과가 다를 때 |
| `VISION_HOST` `VISION_PORT` `VISION_LOG_LEVEL` | `0.0.0.0` `8000` `INFO` | |

비밀은 환경 변수/RunPod 시크릿으로만 넣는다. `docker-compose.yml` 의 `dev-key` 는 로컬 개발 전용이다.

## 구조

설계 §15-3 을 따른다: `src/pair_vision/{api,auth,session,pipeline,models,rules,evidence,health,metrics}`, `configs/`, `tools/replay/`, `ops/runbook.md`, `tests/`. 설계 문서는 pAIr-docs 에 있고 여기엔 링크만 둔다.

## 알려진 한계

계약에서 비어 있거나 해석이 갈려 내가 정한 값은 노트 폴더의 `vision-questions.md` 에 모았다. 브랜치 전략·CI 는 새 저장소 정책 결정 전이라 정하지 않았다.

## 가정한 값 (설계 문서에 값이 없어 보수적으로 둔 임시값 - 모두 `configs/rules.default.yaml`)

| 항목 | 값 | 이유 |
|---|---|---|
| `TABLET` | 20초 연속, 끊김 허용 20초, 쿨다운 30초 | OWLv2 10초 주기(2026-11-27) 샘플 3번 이상(0·10·20초). 끊김 허용은 주기의 2배 이상이어야 한다(설정 검사) |
| `NOTE` (COCO `book`) | 10초 연속, 끊김 허용 3초, 쿨다운 30초 | 프레임마다 검출 |
| `SMARTWATCH` | 30초 연속, 끊김 허용 20초, 쿨다운 60초 | 손목시계를 찬 응시자에게 자주 걸릴 수 있어 더 길게(10초 주기 샘플 4번, 예전 5초 주기 15초와 같은 표본 수) |
| `MANY_HANDS`(손 3개 이상) | 5초 연속, 그림자 모드 | 계약에 유형이 없어 콜백 안 보냄 |
| 검출 기준값 | 키보드 0.3 · 마우스 0.4 · 노트 0.3 · 태블릿/스마트워치/휴대폰 0.5 · 손 0.5 | 문서 값(마우스 0.4, OWLv2 0.5, 손 0.5), 나머지는 측정 때 쓴 값 |
| 손 중복 합침 거리 | 프레임 비율 0.06 | 임시. 8장 사진에서는 중복이 없었다 |

값을 바꾸면 `version` 도 올린다(이벤트 `rulesVersion` 에 기록된다). 사진 8장은 같은 장면의 변형이라 위 값과 검출 정확도를 **확정하지 못한다**.

### 웹캠 규칙의 가정과 한계

- 끊김 허용(`gap_ms`)은 프론트의 0.4초가 아니라 2.5초: 서버는 초당 1장을 분석하므로 프레임 2~3장 간격이다. 지속·쿨다운은 프론트 값 그대로다. 프론트의 "이상이 이어지면 쿨다운 두 배" 반복 경고는 가져오지 않았다(서버는 같은 `clientId` 의 지속시간을 갱신하는 계약).
- 보정은 최소 3장(프론트는 5장). 정면 얼굴 1개가 충분히 안 잡히면 오류 대신 보정을 처음부터 다시 모은다(그동안 `state.calibration` 은 `pending`, 웹캠 이벤트 없음).
- 얼굴·포즈는 프레임마다 둘 다 돌린다(프론트는 얼굴 3번에 포즈 1번 교대). 서버 CPU 부하는 **미측정**.
- 본인 웹캠 사진 6장(정면 4, 고개 숙임 1, 고개 돌림 1; git 에 없음)으로 확인한 것: 얼굴 변환 행렬은 `m[행][열]` 로 읽으면 프론트와 같은 배치다(이동 성분이 `m[:3][3]`, 회전부 det=1). 정면에서 yaw·pitch 가 0 근처, 어깨 검출, 보정 → 이벤트 → 서명 콜백까지 동작한다. 정면 8초 → 고개 숙임 8초 → 정면 흐름에서 `NO_FACE` 가 나고 정면으로 돌아오면 final 로 닫혔다.
- 고개를 약 25° 돌린 사진(yaw +25, 정면 기준 차이 약 28°)으로 `HEAD_TURN` 이 나는 것까지 확인했다(정면 8초 → 돌림 8초: 2.0초 시점 발생, 정면 기준 약 9° 차이인 사진에서는 안 남). 사진은 git 에 없다.
- pitch 부호도 확인했다: 고개를 살짝 숙인 사진에서 pitch +9.9(+ = 숙임, 프론트와 같음), 정면은 약 -1~2. 다만 `HEAD_DOWN` 기준(정면 대비 25°)을 넘는 사진은 얼굴을 놓쳐 확인하지 못했다.
- **못 한 것:** 고개를 많이 숙이거나 돌린 사진은 모델이 얼굴을 놓쳐(IMAGE 모드, 프론트는 VIDEO 추적) `HEAD_DOWN`·`HEAD_TURN` 대신 `NO_FACE` 로 셈된다. `HEAD_DOWN`·`GAZE_AWAY` 가 실제로 **발생**하는 것은 확인하지 못했다(사진 값이 기준값 아래: 시선 X 0.3 안팎 < 0.45).
- 시선 X 부호는 프론트 주석("+ = 응시자 오른쪽")과 **반대**로 나왔다: 응시자 오른쪽(영상 왼쪽)을 보면 -0.32, 응시자 왼쪽(영상 오른쪽)을 보면 +0.27(MediaPipe left/right 이름이 영상 기준으로 보임). 규칙은 X 에 절댓값을 써서 영향이 없다. 시선 Y(+ = 아래)는 눈을 아래로 내린 사진에서 값이 뚜렷하지 않아(-0.04) 부호를 확인하지 못했다. Y 는 부호가 규칙에 직접 쓰이므로(아래만 이상) 고개는 정면으로 두고 눈만 크게 내려다보는 사진이 더 필요하다.

### 시선(`GAZE_AWAY`)은 해상도에 크게 좌우된다 (측정, 본인 사진 1명)

같은 사진을 해상도·화질만 바꿔 `gaze_x`(눈을 영상 왼쪽으로 크게 돌린 사진 / 오른쪽 사진)를 재면:

| 해상도 | 왼쪽 | 오른쪽 | 정면 `gaze_y` 흔들림 |
|---|---|---|---|
| 400px q55 (계약 웹캠 값) | -0.32 | +0.27 | +0.14 |
| 400px q95 | -0.29 | +0.46 | -0.15 |
| 640px q85 | -0.72 | +0.71 | -0.15 |
| 960px q90 | -0.83 | +0.50 | -0.27 |

계약값(400px·q55)에서는 시선 값이 절반 아래로 줄어 기준값 0.45 를 거의 못 넘고, 정면에서도 `gaze_y` 가 사진마다 ±0.15~0.3 흔들린다. **현재 계약 해상도로는 `GAZE_AWAY` 를 믿기 어렵다 → 사용자 결정(2026-11-26)으로 시선 탐지는 하지 않는다(기본 꺼짐, 이벤트 없음. 계약의 유형 목록은 그대로).** 다시 켤 때의 선택지(계약 변경 또는 기준값 재조정이라 이 저장소에서 정하지 않음): 웹캠 `maxEdge` 를 640 안팎으로 올린다(대역폭·100KB 한도 영향 확인 필요), 또는 시선은 그림자 모드로 두고 기준값을 실측으로 다시 정한다. 사람 1명·사진 몇 장이라 확정 수치가 아니다.

### 처리량 개선(워커 병렬 처리, 2026-11-26)
- 병목 실측: 한 프로세스가 프레임을 한 장씩 처리 -> 초당 약 5장(응시자 2명분), GPU 는 놀고 CPU(손·얼굴·포즈, 스레드 경쟁)만 바쁨.
- 대응: `VISION_WORKERS`(프로세스 풀, `models/pool.py`): 응시자 단위 고정 배정·균등 분배, 워커마다 코어 고정(리눅스)과 스레드 수 제한, 죽은 워커만 재기동. 웹캠 포즈는 3프레임마다만 돌리고 사이에는 직전 값을 재사용(`models.pose_every`, CPU 절약).
- **이 PC(CPU 만 있음)에서는 효과를 볼 수 없다**(모델 전체가 CPU 라 워커 1개로도 코어가 다 찬다). 가짜 모델 시험과 진짜 모델 동작(오류 없음)만 확인했다. GPU Pod 에서 노트북의 용량 측정으로 효과를 재야 한다.

### 손·얼굴·몸 모델: RTMW (2026-11-26, MediaPipe 대체)
- **이유:** MediaPipe 는 파이썬에서 CPU 로만 돌아(손 77ms/장 등) 서버 처리량의 병목이었다. RTMW(OpenMMLab, `rtmlib` + ONNX Runtime)는 손·얼굴 랜드마크·몸(어깨) 133점을 **GPU 한 모델**로 낸다. 기본 백엔드(`models.vision_backend: rtmw`), MediaPipe 는 `mediapipe` 로 바꾸면 그대로 쓸 수 있다(`pip install .[mediapipe]` + `.task` 모델 필요).
- **모델 파일**(`models/rtmw/`, lock 에 sha256): `yolox_tiny.onnx`(사람 검출, 18.9MB zip 에서 추출), `rtmw-dw-l-m_256x192.onnx`(133점, 120MB zip 에서 추출). 출처 download.openmmlab.com, Apache-2.0(코드·가중치 표기). **학습 데이터(COCO-WholeBody 등)의 이용 조건은 확인하지 못했다 - 법무 확인 대상.**
- **휴대폰:** 프레임 전체를 사람 한 명의 상자로 넣어 손 두 개를 얻는다(사람 검출기를 거치면 팔이 사람 여러 명으로 잡혀 중복됐다). 책상 사진 8장 손 개수 `2,2,2,2,2,2,1,1` 이 눈으로 센 값(5번만 3개)과 맞는다. **한 상자에서는 손이 둘까지라 "손 3개 이상"은 감지하지 못한다**(이벤트 유형을 만들지 않기로 해서 영향은 작다).
- **웹캠:** 사람 검출 -> RTMW. 얼굴 수는 얼굴 점수가 `face_threshold` 이상인 사람 수. 머리 각도는 얼굴 6점 PnP 로 새로 계산(yaw 는 응시자 오른쪽이 +, pitch 는 숙임이 +). **본인 웹캠 사진 10장에서 RTMW 는 고개를 숙이거나 많이 돌린 사진에서도 얼굴을 잡았다**(MediaPipe 는 놓쳐 `NO_FACE`). 숙인 사진에서 `HEAD_DOWN`(6초 지속) 이 발생하는 것을 확인했다.
- **임시 보정값:** RTMW 6점 모델의 yaw 는 실제보다 2배쯤 크게·흔들려 `yaw_scale: 0.55`(MediaPipe 값과의 비교, **한 사람 10장 기준**). 다른 사람·자세로 재확인이 필요하다. 시선(`GAZE_AWAY`)은 계속 끈다.
- **GPU:** ONNX Runtime GPU(CUDA) 로 돌려야 효과가 난다. bootstrap.sh 가 GPU 가 있으면 `onnxruntime-gpu` 로 교체한다. **GPU Pod 에서는 아직 실행해 보지 못했다**(이 PC 는 CPU). CUDA 를 못 잡으면 CPU 로 돌고 `/readyz` 의 `gpu` 가 `false` 로 나온다.
- 속도는 이 PC(CPU)에서 프레임당 약 0.15~0.35초(YOLOX+RTMW)였다. GPU 수치는 Pod 에서 용량 측정으로 확인한다.

### GPU 시간 줄이기 (2026-11-26, 워커 6개로도 4명 OK·8명 부족이었던 실측 뒤)
- **가설:** 응시자 1명당 GPU 시간이 D-FINE(프레임마다 약 97ms) + OWLv2(5초마다 155ms) + RTMW 로 초당 약 0.17초라 GPU 하나가 약 6명에서 포화된다(계산이고 단계별 실측은 이 변경 뒤에 나온다). 프로세스를 늘려도 GPU 시간은 나눠 쓸 뿐이다.
- **D-FINE 주기화(`models.dfine_interval_ms: 3000`):** 책상 위 키보드·마우스·노트는 거의 안 움직여서(설계 §7-3 "정지 박스 캐시") 세션마다 3초에 한 번만 돌리고 사이에는 직전 결과를 재사용한다. 노트 이벤트의 끊김 허용을 3초에서 7초로 늘렸다(D-FINE 주기의 2배 이상, 설정 검사). 0 이면 예전처럼 매 프레임.
- **단계별 시간 측정:** `/metrics` 에 `pair_vision_stage_seconds_total{stage}`·`..._calls_total{stage}`(decode·dfine·hands·owlv2·webcam). 노트북 용량 측정이 단계마다 평균 ms 와 "초당 점유초"를 표로 낸다. 점유 합이 워커 수에 가까우면 모델 계산이 병목이다.
- 아직 안 한 것(필요하면 이 순서): D-FINE·OWLv2 반정밀도(fp16), 세션을 묶은 GPU 배치 추론, OWLv2 주기 늘리기, 전송 간격 줄이기, 서버 2대.

### 묶음 추론·fp16·OWLv2 10초 (2026-11-27, GPU 1장 20명 목표 - `docs/intent/realtime-vision-20.md`)
- **왜:** 용량 시뮬레이션에서 워커 풀 구조는 GPU 1장 약 8명이었다. 병목은 GPU 모델 시간(모델 밖 서버는 100명도 폐기 0).
- **묶음 추론(`VISION_BATCH=1`, `models/batch.py`·`batch_real.py`):** 모델을 한 프로세스에 두고, 프레임마다 단계 목록(휴대폰: [D-FINE] → RTMW 손 → [OWLv2], 웹캠: [YOLOX 사람 상자] → RTMW)을 정한 뒤 **엔진 스레드 1개**가 "가장 오래 기다린 작업이 있는 단계"의 대기 작업을 최대 16장(OWLv2 4장) 한 번에 돌린다. 휴대폰 손과 웹캠 얼굴·몸은 같은 RTMW 묶음에 섞인다(RTMW ONNX 는 배치 축이 있고 2의 거듭제곱으로 채워 입력 모양을 고정). YOLOX ONNX 는 배치 1 고정이라 한 장씩.
- **"묶음이 다 끝나야 다음 묶음" 구조를 없앴다(`pipeline/worker.py` 연속 처리):** 칸(응시자·채널)마다 처리 중 1장만 두고, 프레임이 끝나는 즉시 그 칸의 다음 프레임을 넘긴다. 빠른 웹캠 프레임이 느린 OWLv2 묶음을 기다리지 않는다. 큐 길이 1·seq·규칙 엔진·콜백 계약은 그대로.
- **fp16(`models.fp16`, GPU 일 때만):** OWLv2 는 가중치까지 fp16, **D-FINE 은 혼합 정밀도(autocast)**. D-FINE 을 통째로 `.half()` 하면 이 PC 측정에서 책상 사진 3장의 키보드·마우스·노트가 전부 사라졌다(값 넘침으로 보임). autocast 는 fp32 와 검출이 같았다(점수 차 0.005). fp16 출력에 NaN/inf 가 나오면 그 프레임을 오류로 버린다. 모델별로 끄려면 `VISION_FP16_SKIP`.
- **전처리 전부 GPU(`VISION_DEVICE_PREPROCESS`, GPU 면 자동):** CPU 는 단계(주기) 판단만 하고, JPEG 풀기(torchvision nvJPEG, 묶음 한 번 - torchvision 이 없으면 PIL 로 풀어 올림)·D-FINE/OWLv2 전처리·RTMW 자르기(rtmlib 아핀 행렬 그대로 grid_sample)·YOLOX 레터박스를 엔진 스레드에서 GPU 로 한다. rtmlib·PIL 과 수치 비교 시험, CPU 강제 실행에서 기존 경로와 검출 결과 같음. 모델 입출력(ONNX Runtime)은 호스트 numpy 를 거치고(작은 복사), RTMW 점 위치 계산·규칙 판정은 CPU(µs 단위). 시뮬레이션상 GPU 가 병목이라 용량은 약 1명 줄어든다(26 -> 25명 추정).
- **빠른 전처리(`models.fast_preprocess`):** D-FINE·OWLv2 전처리를 torch 로(GPU 면 GPU 에서). OWLv2 기준 전처리(scipy) 이 PC CPU 132ms → torch 1스레드 26ms, D-FINE 12ms → 5ms. 기준 처리기와 수치 비교 시험(`tests/test_fast_preprocess.py`, 평균 차 < 0.005).
- **OWLv2 10초 주기 + 태블릿·스마트워치 규칙값**(위 "가정한 값"), **웹캠 사람 상자 재사용**(`models.person_box_interval_ms: 3000`, 사이에는 직전 상자로 RTMW. 새 사람은 최대 3초 늦게 잡힌다). 규칙 `version` 2026-11-27.1.
- **지표:** `pair_vision_batch_size{stage}`(묶음 크기), `pair_vision_stage_seconds_total`·`..._calls_total`(묶음에서는 호출 수 = 프레임 수라 초/호출 = 프레임당 평균), `pair_vision_processed_gap_seconds{channel}`(같은 칸에서 분석된 프레임 사이 간격 - 끊김 허용 초과 비율), `pair_vision_inflight_frames`, 지연 버킷에 3초 추가.
- **시뮬레이션(추정, `tools/capacity_sim/capacity_sim.py` ⑦):** 구현한 구조로 GPU 1장 약 25명(합격 기준: 폐기 1%·p95 3초·끊김 1%), 20명에서 폐기 0%·p95 0.75초·GPU 점유 99%. 묶음 이득이 없거나(f8=1.0) RTMW 가 40ms 이상이면 20명 불합격 - **GPU 실측이 필요하다**: `tools/gpu_test/`(합격 시험 노트북, runbook §6-5).
- **검증:** 가짜 모델·실제 모델(CPU) 둘 다 묶음 결과가 한 장씩 결과와 같음(`tests/test_batch_real.py`), 실제 모델 묶음 서버 + 부하 3명(CPU, 콜백 서명 검증)까지. GPU 실측은 아래 절.

### GPU 합격 시험 실측 (2026-11-27, RunPod RTX A4500 20GB · CPU 48코어 - 합격)
`tools/gpu_test/` 노트북(runbook §6-5) 한 번 실행. 조건: 응시자 20명 x 웹캠·휴대폰 x 초당 1장 x 20분(앞 1분 제외), 합성 사진, `VISION_BATCH=1`, 규칙 2026-11-27.1, 부하는 Pod 안 127.0.0.1.

| 합격 기준 | 실측 | 한도 |
|---|---|---|
| 버린 사진 | 0 / 47,994장 (0%) | 1% |
| 경고 지연 p95 | 0.10초 (100% 가 3초 안) | 3초 |
| 끊김 초과 휴대폰(>3초) / 웹캠(>2.5초) | 0% / 0% (간격 p95 1.37초) | 1% |

- 정합성: `/readyz` gpu=true, 모델 sha256 lock 대조 통과, 모델 오류 0, 콜백 실패 0, 서명 검증 수신기에 이벤트 107건(시작·끝 합쳐 184번 전송) 도착.
- 정밀도: D-FINE 혼합 정밀도(autocast fp16), OWLv2 fp16. fp32 대비 최대 점수 차 0.005 / 0.002, 짝 없는 검출 0 -> fp32 로 둔 모델 없음. 전처리 전부 GPU(nvJPEG 풀기 포함).
- 자원: GPU 사용률 평균 17%·최대 63%, GPU 메모리 최대 3.4GB, load1 평균 2.58, CPU 일 프레임당 0.01ms.
- 프레임당 시간(20명, 평균 묶음 1~2장): D-FINE 33ms(3초 주기) · OWLv2 32ms(10초 주기) · RTMW 4.9ms · YOLOX 9.5ms(3초 주기) · JPEG 풀기 0.6ms. 엔진 점유 합 약 0.55(대부분 GPU 계산 밖의 파이썬·동기화).
- 묶음 속도 곡선(장당 ms): D-FINE 1장 27 -> 16장 4.4(배수 0.16) · RTMW 1장 4.7 -> 32장 1.3(0.27) · OWLv2 1장 31 -> 4장 24(0.78) · YOLOX 7.7(배치 1 고정, 이득 없음). 시뮬레이션 가정(배수 0.4, RTMW 25ms)보다 대체로 빨라 시뮬레이션의 25명 추정은 보수적이었다.
- **결론(의도 파일 기준): GPU 1장 20명 실시간 감독 유지, 사후 분석 전환 안 함.**
- 확인 못 한 것: 20명 초과(실측 없음 - 곡선상 여유는 크지만 묶음마다 드는 파이썬 비용·YOLOX 가 먼저 막을 수 있다. 늘리려면 노트북 `STUDENTS` 만 바꿔 다시 잰다), 실제 사진, RunPod 프록시를 거친 WebSocket 수신, 실제 백엔드 연결.
