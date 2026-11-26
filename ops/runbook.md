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

## 6. 팀원용: RunPod 에 실제 모델 올려 보기 (GPU 는 아직 한 번도 안 돌려 봤다)
1. **이미지 만들기**(이 PC 에서 CPU 로 빌드·실행은 확인됨, 19.6GB): 저장소 `main` 에서
   `docker build --build-arg UV_EXTRAS="--extra real" -t <레지스트리>/pair-vision:real .`
   Linux 용 PyPI torch 에 CUDA 라이브러리가 들어 있어서 기본 베이스(`python:3.12-slim`)로도 GPU 가 잡힐 것으로 본다(**미검증**). 안 잡히면 `--build-arg BASE_IMAGE=<CUDA 런타임+Python 3.12 이미지>` 를 준다.
2. **레지스트리에 올리기**: `docker push <레지스트리>/pair-vision:real` (용량이 커서 회선이 느리면 빌드·푸시를 빠른 네트워크 머신이나 CI 에서 한다).
3. **가중치 준비**: RunPod 네트워크 볼륨(또는 Pod 볼륨)에 `models/` 와 같은 구조로 5개를 둔다: `dfine-small-coco/` `owlv2-base-patch16-ensemble/` `hand_landmarker.task` `face_landmarker.task` `pose_landmarker_lite.task` (sha256 은 `configs/models.lock`, 시작할 때 자동 검증; 파일은 git 에 없으니 이 PC 의 `models/` 를 복사해 간다).
4. **Pod 만들기**(RunPod 화면 항목 이름은 확인 필요): 위 이미지, GPU 1장(측정은 RTX 4000 Ada 20GB), 볼륨을 `/models` 에 마운트, HTTP 포트 8000 공개, 환경 변수:
   `VISION_DETECTOR=real` `VISION_MODELS_DIR=/models` `VISION_API_KEY=<비밀>` `VISION_CALLBACK_URL=<백엔드 /internal/vision/events>` `VISION_CALLBACK_SECRET=<비밀>` (`VISION_CALLBACK_KEY_ID` 기본 k1). 비밀은 RunPod 시크릿으로.
5. **확인**: 모델 적재에 수십 초. `GET /healthz`(인증 없음) → 200, `GET /readyz`(Bearer `VISION_API_KEY`) → `modelsLoaded:true` **`gpu:true`**. `gpu:false` 면 GPU 가 안 잡힌 것(로그 확인, 위 1번 베이스 이미지 대안).
6. **재생 시험**(백엔드 없이): 이 PC 에서 `tools/replay/replay.py --url https://<pod 주소> --api-key <키> --photos <사진 폴더> --photo-names 1.png --duration 30` 를 돌려 `/metrics` 의 `pair_vision_frame_latency_seconds`(OWLv2 가 있는 프레임 포함)와 이벤트 수를 본다. 콜백은 백엔드 대신 `tools/replay/sink.py` 주소로 받아 서명 검증을 확인할 수 있다.
7. **하지 않는 것**: 백엔드 스위치는 켜지 않는다(로컬 시험 → 배포 테스트 → 롤백 순서, 설계 §14). 사진·키·콜백 비밀을 로그나 채팅에 남기지 않는다.

### 6-1. 노트북으로 한 번에 (총괄 노트북, 저장소 밖 파일)
RunPod PyTorch 템플릿 Pod 의 JupyterLab 에 총괄 노트북을 복사해 위에서부터 실행한다: 코드 받기(`git clone`, 비공개면 토큰을 물어봄) → `pip install -e ".[real]"` → 모델 5개 내려받기(Hugging Face·Google)와 `models.lock` sha256 검증 → 서버 실행 → `gpu` 확인 → 재생 시험(사진이 없으면 합성 이미지로 속도만) → 종료. **노트북은 GitHub 의 `main` 을 받으므로 먼저 이 저장소를 푸시해 둬야 한다.** 사진·응시자 실사진은 노트북에 들어 있지 않다. GPU 에서는 아직 돌려 보지 못했다.

### 6-2. 백엔드 연결 (백엔드 `feat/server-vision` 의 `application.yml` 확인 결과)
| 백엔드 환경 변수 | 넣을 값 (서버 쪽 대응) |
|---|---|
| `PAIR_VISION_INFERENCE_URL` | 추론 서버 주소 (RunPod HTTP 프록시 `https://<pod id>-8000.proxy.runpod.net`) |
| `PAIR_VISION_INFERENCE_KEY` | 서버 `VISION_API_KEY` 와 같은 값 (Bearer) |
| `PAIR_VISION_EVENT_HMAC_SECRET` (`..._NEXT` 는 교체용) | 서버 `VISION_CALLBACK_SECRET` 과 같은 값. 서버 `VISION_CALLBACK_KEY_ID` 는 `k1`(`..._NEXT` 는 `k2`) |
| `PAIR_SERVER_VISION_ENABLED` | **`false`** 로 두고 시작(테스트 회차에서만 켠다) |
- 서버 → 백엔드 콜백 주소 `VISION_CALLBACK_URL` 은 백엔드의 `/internal/vision/events` 전체 주소(RunPod 에서 닿는 공개 주소).
- **현재 백엔드에는 실제 추론 서버 클라이언트(`/v1/stream`·REST 호출)가 없고 가짜 `StubInferenceClient` 만 있다**(`infra/ai/InferenceClient.java` 주석). 그 구현이 들어가기 전에는 백엔드↔서버 종단 연결이 불가능하다. 그 전까지의 확인은 노트북 6번 칸(서버 단독 재생 시험)과 콜백은 `tools/replay/sink.py` 로 한다. RunPod 프록시가 WebSocket 을 통과시키는지는 노트북 7번 칸이 확인한다.

### 6-3. Pod 를 켤 때마다 라이브러리·모델을 다시 받지 않기 (영구 볼륨 + 스크립트)
- **원인**: Pod 를 다시 켜면 컨테이너 디스크가 초기화되어 apt 로 깐 시스템 라이브러리(libEGL·libxcb 등)와 pip 로 깐 패키지가 사라진다. 영구 볼륨(`/workspace`)만 남는다. **Pod 를 만들 때 Volume(영구 저장소)을 반드시 붙인다.**
- **해결**: `tools/runpod/bootstrap.sh` (멱등): ① 빠진 시스템 라이브러리만 apt 로 설치(수 초) ② `/workspace/venv-vision` 가상환경(Pod 의 torch 재사용)과 ③ 모델 5개를 영구 볼륨에 한 번만 설치·검증하고 다음부터 건너뜀(`pyproject.toml`·`uv.lock` 이 바뀌면 재설치). 리눅스 컨테이너에서 "처음 설치 약 5분 → 새 컨테이너로 재기동 → 가상환경·모델 건너뛰고 서버 기동(readyz `modelsLoaded:true`)"까지 확인했다. GPU 는 미검증.
- **실행 방법 두 가지**
  1. **총괄 노트북**(저장소에 넣지 않고 별도 파일로 전달해 RunPod 의 Jupyter 에 복사해 쓴다. 노트북이 저장소를 스스로 `git clone`/`git pull` 한다): Pod 를 켤 때마다 `Run All` 한 번이면 코드 받기/갱신 → 환경 준비(bootstrap.sh) → 서버 실행 → 외부 접속 확인 → 백엔드 값 출력까지 한다. 키·콜백 정보는 처음 한 번만 묻고 `/workspace/.pair-vision.env`(본인만 읽기)에 저장해 다음부터 같은 값을 쓴다. 사진 재생 시험(`RUN_TEST`)과 서버 끄기(`STOP`)는 스위치를 바꿔야 실행된다.
  2. 노트북 없이: Pod 터미널에서 `git clone` 후 `VISION_API_KEY=<키> bash tools/runpod/start.sh` (RunPod 템플릿의 Container Start Command 로 `bash /workspace/pAIr-vision/tools/runpod/start.sh` 를 줘도 된다. 콜백을 쓰면 `VISION_CALLBACK_URL`·`VISION_CALLBACK_SECRET` 도 환경 변수로).
- 코드를 갱신하려면 `git -C /workspace/pAIr-vision pull` 후 다시 실행. 새 모델이나 패키지 변경은 위 스크립트가 알아서 반영한다.
- 이 방식도 첫 기동은 느리다(패키지 수 GB + 모델 약 700MB). 첫 기동을 없애려면 도커 이미지에 다 넣어 레지스트리에 올리는 방식(§1·§6)이 더 낫다.

### 6-4. 압축 파일(구글 드라이브) 방식 - 권장 (저장소·토큰 불필요)
- 서버 코드 + 모델 5개를 `pair-vision-bundle.zip`(약 716MB)으로 묶어 구글 드라이브('링크가 있는 모든 사용자')에 올리고, 총괄 노트북(저장소 밖 파일)과 함께 전달한다. 노트북이 링크로 받아 **크기·sha256 을 검증**하고 풀어서 쓴다(모델은 zip 안에 있어 따로 내려받지 않는다). zip 을 JupyterLab 에 직접 올려도 된다(파일 이름 `pair-vision-bundle.zip`).
- 코드·가상환경·모델은 `/workspace` 가 아니라 홈 아래 `~/pair-vision-work` 에 둔다(`/workspace` 권한 문제 회피). 서버 키 같은 작은 설정 파일만 `/workspace/.pair-vision/env.json` 에 저장한다. Pod 를 새로 만들면 압축 해제와 파이썬 패키지 설치를 다시 한다.
- 새 버전을 배포하려면 zip 과 노트북을 다시 만들어 교체한다(zip 의 크기·sha256 이 노트북에 박혀 있어 둘은 짝이다).
- RunPod PyTorch 템플릿의 torch 가 2.6 미만이면 `bootstrap.sh` 가 가상환경에 torch 2.6(CUDA 맞춤)을 추가로 설치한다.
