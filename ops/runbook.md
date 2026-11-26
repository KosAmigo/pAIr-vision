# 시험 당일 런북 (골격)

> 설계 §10·§11 의 절차를 채워 넣는 자리. 아래 **TODO** 는 실측·결정 뒤에 채운다.

## 0. 시험 전날
- [ ] 백엔드 `PAIR_SERVER_VISION_ENABLED` 와 대상 회차 `server_vision` 확인
- [ ] 키 교체가 필요하면 `VISION_API_KEY_NEXT` 로 새 키를 먼저 올리고, 백엔드를 새 키로 바꾼 뒤 옛 키를 내린다
- [ ] 콜백 비밀 키 번호(`k1`/`k2`) 가 백엔드와 같은지 확인
- [ ] 규칙 값(`configs/rules.default.yaml`)과 `version` 확인

## 1. 시험 시작 전 (RunPod 기동)  — 이미지·환경 변수는 확정, RunPod 템플릿·기동 API 는 TODO(설계 §11-1)
0. 이미지: `docker build --build-arg UV_EXTRAS="--extra real" --build-arg BASE_IMAGE=<CUDA 런타임+Python 3.12> -t pair-vision:gpu .` (**아직 빌드해 보지 못함**). 가중치 5개(`dfine-small-coco/` `owlv2-base-patch16-ensemble/` `hand_landmarker.task` `face_landmarker.task` `pose_landmarker_lite.task`)는 RunPod 네트워크 볼륨에 두고 `VISION_MODELS_DIR` 로 가리킨다. sha256 은 `configs/models.lock` 과 시작할 때 대조된다.
0-1. 환경 변수: `VISION_DETECTOR=real`, `VISION_MODELS_DIR`, `VISION_API_KEY`, `VISION_CALLBACK_URL`, `VISION_CALLBACK_SECRET`(RunPod 시크릿). 모델 적재에 수십 초 걸리므로 시험 30분 전에 올린다.
1. 인스턴스 기동, 컨테이너 시작
2. `GET /healthz` 200, `GET /readyz` 200 (`modelsLoaded`, `gpu`, `maxSessions`, `rulesVersion` 확인)
3. 백엔드가 `/readyz` 를 5초마다 호출하는지(연속 3회 실패 시 제외) 확인
3-1. 실제 모델이면 `/readyz` 의 `modelsLoaded:true`, `gpu`, `rulesVersion` 확인. 모델 폴더 sha256 이 `configs/models.lock` 과 다르면 서버가 시작하지 않는다(로그의 `모델 검증 실패`). 가중치는 이미지에 넣지 않고 `VISION_MODELS_DIR` 볼륨에 둔다.
4. 시험 직전 리허설: `tools/replay/replay.py` 로 몇 스트림을 흘려 콜백이 백엔드까지 가는지 확인

## 2. 시험 중 감시  — TODO(Grafana 위치 미정)
- 지표(`/metrics`): `pair_vision_sessions`, `pair_vision_frames_dropped_total`, `pair_vision_frame_latency_seconds`, `pair_vision_callback_queue_size`, `pair_vision_callback_failures_total`, `pair_vision_gpu_utilization_ratio`
- 경고 기준(폐기율·지연·GPU)은 M5 실측 뒤 정한다. 콜백 대기 큐가 계속 늘면 백엔드 `/internal/vision/events` 장애를 의심한다.
- 장애 시: 컨테이너 재시작 → 백엔드가 활성 세션에 `start(recalibrate:true)` 를 다시 호출한다(계약 §3-6). 재시작한 동안은 백엔드가 감독 중단으로 기록한다.

## 3. 교체·종료
- 무중단 교체: `POST /v1/admin/drain` → `/readyz` 의 `sessions` 가 0 이 될 때까지 대기 → 종료 (drain 중에도 `/readyz` 는 200 이고 `draining:true`)
- 시험 종료: 백엔드가 각 세션에 `end` 를 호출했는지(`sessions` 0) 확인 후 인스턴스를 내린다
- 종료 전 콜백 대기 큐(`pair_vision_callback_queue_size`)가 0 인지 확인 — 남아 있으면 마지막 이벤트가 유실된다

## 4. 시험 뒤
- [ ] 로그에 사진·키·서명이 없는지 샘플 점검 (접근 로그는 꺼져 있어야 함)
- [ ] 폐기율·지연·오탐 기록을 §12 측정 표에 옮김

## 5. 실제 모델 로컬 실행 (이 PC, CPU)
- README 의 "실제 모델로 실행" 절 순서대로: 콜백 수신기(`tools/replay/sink.py`) → `VISION_DETECTOR=real` 서버 → `tools/replay/replay.py --photos <폴더>`.
- CPU 에서는 `VISION_RULES_PATH=configs/rules.local-cpu.yaml` 을 쓴다(OWLv2 주기 30초). 운영 파일이 아니다.
- 웹캠 채널은 `replay.py --channel webcam` 으로 흘린다(얼굴이 나온 사진이 있어야 보정이 끝나고 이벤트가 난다).
- 확인할 것: `/readyz` `modelsLoaded`, sink 에 서명 검증된 `HANDS_OUT`·`HAND_OUT_ONE`·`TABLET`·`NOTE`·`SMARTWATCH`, `/metrics` 의 `pair_vision_shadow_events_total{type="MANY_HANDS"}`(그림자).
- 롤백: 서버 모델 문제는 백엔드 스위치를 끄는 것으로 되돌린다(설계 §14-3). 서버가 죽어도 시험은 계속되고 감독 중단으로 기록된다.
