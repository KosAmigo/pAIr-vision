# pAIr-vision

pAIr 온라인 시험 서비스의 **서버 추론 감독용 추론 서버**(RunPod GPU 에서 돌릴 예정). 백엔드(`pAIr-back`)가 보낸 웹캠·휴대폰 사진을 최신 한 장씩 분석해, 규칙에 걸린 이상 상태를 서명된 HTTP 콜백으로 백엔드에 알린다.

- 계약: pAIr-docs `설계/서버_추론_감독_설계.md` §3·§6·§9 (사본 `contract.md`). **계약을 바꾸려면 문서 담당 세션을 통해 고친다.**
- 이 저장소의 현재 범위: **계약 뼈대 + 가짜 모델**. 실제 모델·라이선스(D-6)는 정해지지 않아 넣지 않았다(`src/pair_vision/models/` 의 `Detector` 인터페이스만 있다).
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

웹캠 규칙(`NO_FACE` 등 pcRules 포팅)과 `TABLET`·`NOTE`·`SMARTWATCH`·`PHONE_MOVED` 는 모델이 정해진 뒤의 일이라 아직 없다.

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

콜백 수신기와 부하 재생기까지 보려면:

```powershell
docker compose --profile demo up -d --build
uv run python tools/replay/replay.py --api-key dev-key --streams 3 --scenario "NORMAL:1,BOTH_OUT:12" --duration 12
docker compose logs sink      # 서명이 검증된 HANDS_OUT 이벤트
```

60스트림 부하는 `VISION_MAX_SESSIONS=60` 으로 띄우고 `--streams 60`.

## 환경 변수

| 이름 | 기본 | 설명 |
|---|---|---|
| `VISION_API_KEY` / `VISION_API_KEY_NEXT` | (필수 / 없음) | 백엔드 Bearer 키, 무중단 교체용 두 개 |
| `VISION_CALLBACK_URL` | 없음 | 백엔드 `/internal/vision/events` 전체 주소. 없으면 이벤트를 보내지 않고 버린 수만 센다 |
| `VISION_CALLBACK_KEY_ID` / `VISION_CALLBACK_SECRET` | `k1` / (URL 이 있으면 필수) | 서명 키 번호와 비밀 |
| `VISION_MAX_SESSIONS` | 40 | 동시 세션 상한 (**실측 M4·M5 전 임시값**) |
| `VISION_RULES_PATH` | `configs/rules.default.yaml` | 규칙 파일 |
| `VISION_DETECTOR` / `VISION_FAKE_SCENARIO` | `fake` / `normal` | 모델·가짜 시나리오(`hands_out` `one_hand_out` `flicker` 또는 `NORMAL:3,BOTH_OUT:10,NORMAL:1`) |
| `VISION_HOST` `VISION_PORT` `VISION_LOG_LEVEL` | `0.0.0.0` `8000` `INFO` | |

비밀은 환경 변수/RunPod 시크릿으로만 넣는다. `docker-compose.yml` 의 `dev-key` 는 로컬 개발 전용이다.

## 구조

설계 §15-3 을 따른다: `src/pair_vision/{api,auth,session,pipeline,models,rules,evidence,health,metrics}`, `configs/`, `tools/replay/`, `ops/runbook.md`, `tests/`. 설계 문서는 pAIr-docs 에 있고 여기엔 링크만 둔다.

## 알려진 한계

계약에서 비어 있거나 해석이 갈려 내가 정한 값은 노트 폴더의 `vision-questions.md` 에 모았다. 브랜치 전략·CI 는 새 저장소 정책 결정 전이라 정하지 않았다.
