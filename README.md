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

> 실제 모델 도커 이미지는 **빌드·실행해 보지 못했다**(이 PC 에서 torch 이미지를 받지 않음). RunPod GPU 로 올리기 전에 한 번 빌드해 `/readyz` 의 `modelsLoaded`·`gpu` 를 확인해야 한다.

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
| `VISION_HOST` `VISION_PORT` `VISION_LOG_LEVEL` | `0.0.0.0` `8000` `INFO` | |

비밀은 환경 변수/RunPod 시크릿으로만 넣는다. `docker-compose.yml` 의 `dev-key` 는 로컬 개발 전용이다.

## 구조

설계 §15-3 을 따른다: `src/pair_vision/{api,auth,session,pipeline,models,rules,evidence,health,metrics}`, `configs/`, `tools/replay/`, `ops/runbook.md`, `tests/`. 설계 문서는 pAIr-docs 에 있고 여기엔 링크만 둔다.

## 알려진 한계

계약에서 비어 있거나 해석이 갈려 내가 정한 값은 노트 폴더의 `vision-questions.md` 에 모았다. 브랜치 전략·CI 는 새 저장소 정책 결정 전이라 정하지 않았다.

## 가정한 값 (설계 문서에 값이 없어 보수적으로 둔 임시값 - 모두 `configs/rules.default.yaml`)

| 항목 | 값 | 이유 |
|---|---|---|
| `TABLET` | 10초 연속, 끊김 허용 10초, 쿨다운 30초 | OWLv2 5초 주기 샘플 3번 이상. 끊김 허용은 주기의 2배 이상이어야 한다(설정 검사) |
| `NOTE` (COCO `book`) | 10초 연속, 끊김 허용 3초, 쿨다운 30초 | 프레임마다 검출 |
| `SMARTWATCH` | 15초 연속, 끊김 허용 10초, 쿨다운 60초 | 손목시계를 찬 응시자에게 자주 걸릴 수 있어 더 길게 |
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
