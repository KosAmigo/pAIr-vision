# 측정용 사진 수집 가이드

목적: `tools/bench/compare.py` 로 모델을 같은 사진에 돌려 비교한다. 정답 라벨이 없으면 속도와 "무엇을 찾았나"만 볼 수 있고 정확도는 못 낸다. 정확도를 보려면 사진마다 정답(물체 위치·손 안/밖)을 사람이 따로 붙여야 한다.

## 촬영 규칙

- **누구**: 동의한 사내 인원만(설계 §8-3). 실제 응시자 사진은 쓰지 않는다. 얼굴이 나오면 보관 기간·접근 권한을 정한다(D-8).
- **휴대폰 거치**: 책상 옆 거치대, 키보드·마우스·두 손이 모두 화면에 들어오게. 각도는 3가지(낮게/중간/높게)를 사람마다 바꿔 찍는다. 세로/가로 방향도 응시자가 실제로 쓸 방향으로.
- **해상도**: 서버로 보낼 크기 두 가지로 각각 저장 — 긴 변 512px 와 1280px (설계 §8-4). 원본도 따로 보관.
- **조명**: 밝은 실내 / 어두운 실내(스탠드만) / 역광(창가) 3가지.
- **환경 변형**: 노트북·데스크톱, 키보드 종류(검정·흰색·기계식), 마우스 종류, 책상 색.
- **분리**: 사람·환경 단위로 폴더를 나눠 평가용과 학습용이 섞이지 않게(`photos/<사람>/<환경>/*.jpg`). 도구는 하위 폴더까지 읽는다.

## 찍을 장면

| 번호 | 장면 | 목적 |
|---|---|---|
| 1 | 정상: 양손 키보드·마우스 | 키보드·마우스·손 검출 기준 |
| 2 | 두 손 모두 영역 밖(무릎·턱·머리) | 손 규칙 |
| 3 | 한 손만 밖(다른 손은 영역 위) | 손 규칙 |
| 4 | 손이 화면 밖으로 사라짐 | 설계 §7-2-1 |
| 5 | 태블릿을 책상에 둠(켜짐/꺼짐) | TABLET |
| 6 | 메모지·포스트잇·책 | NOTE |
| 7 | 스마트워치 착용(손목 보임/소매에 가림) | SMARTWATCH |
| 8 | 간섭 물체: 컵·리모컨·손목시계·휴대폰·이어폰 케이스 | 오탐 |

필요한 물체 목록: 키보드, 마우스, 태블릿, 메모지/포스트잇/책, 스마트워치, 컵, 리모컨, 일반 손목시계.
장면당 1~2분 영상에서 초당 1~2장 추출(설계 §8-3). 클래스당 300~500장은 직접 학습 단계의 출발 규모이며, 단순 속도·동작 확인은 20~50장이면 된다.

## 측정 실행

```powershell
uv run python -m tools.bench.compare --images C:\photos\desk --models fake
uv run python -m tools.bench.compare --images C:\photos\desk --models rtdetr-r18,dfine-s,owlv2,mediapipe-hands --json out.json
```

가중치·라이브러리가 없는 모델은 건너뛴다. GPU(CUDA) 없는 PC의 수치는 CPU 기준이라 RunPod 용량(M4)으로 환산할 수 없다.

## 정답 라벨 붙이기 (정확도 평가용)

```powershell
uv run python -m tools.eval.evaluate template --photos C:\photos --out labels.csv   # 사진마다 한 줄인 빈 CSV
```

- 열: `file`(파일 이름) `channel`(`phone`/`webcam`, 폴더 이름에 webcam 이 있으면 자동) `keyboard` `mouse` `phone` `tablet` `note` `smartwatch`(사진에 **보이면 1, 안 보이면 0**) `hands`(보이는 손 개수) `persons`(사람 수) `faces`(얼굴 수).
- 칸을 비우면 그 사진·항목은 평가에서 빠진다. 헷갈리는 사진은 비워 둔다(애매한 정답이 점수를 흐린다).
- 일부만 가려진 물체(손에 반쯤 가린 휴대폰 등)는 사람이 봐서 알아볼 수 있으면 1.
- 평가: `uv run python -m tools.bench.compare --images C:\photos --models dfine-s,owlv2,mediapipe-hands --hand-conf 0.5,0.3,0.1 --json out.json` 으로 결과를 만들고 `uv run python -m tools.eval.evaluate score --truth labels.csv --results out.json --thresholds 0.2,0.3,0.4,0.5,0.6,0.7 --out eval`. 기준값을 한 번에 훑어 클래스별 precision/recall 표(`eval.md`, `eval.csv`)가 나온다.
- 빈 양식: `tools/eval/labels.template.csv`.
