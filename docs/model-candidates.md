# 로컬 모델 비교: 환경과 후보 모델 (2026-11-25 조사)

## 이 PC 환경

| 항목 | 값 |
|---|---|
| OS | Windows 11 Home 10.0.26300 |
| CPU / RAM | Intel Core Ultra 7 155H (16코어) / 31.6GB |
| GPU | **Intel Arc Graphics(내장, 공유 메모리)** — NVIDIA 아님, `nvidia-smi` 없음, **CUDA 없음** |
| Python / uv | 3.14.8 / 0.13.10 |
| 디스크 여유 | C: 약 685GB |

**결론: 이 PC의 측정은 전부 CPU 기준**이다. ms/장 수치는 RunPod GPU 성능이 아니며 설계 §12 M4(GPU 1장당 스트림 수)를 대신하지 못한다. 모델끼리의 상대 비교·동작 확인용이다.
측정 환경은 `.venv-bench`(Python 3.12.16, torch 2.15 CPU, transformers 5.24, mediapipe; git 제외)에 설치했다. 실행: `$env:PYTHONPATH="src;."; .\.venv-bench\Scripts\python.exe -m tools.bench.compare --images <폴더> --models ...`. 합성 사진 3장으로 5개 어댑터가 끝까지 도는 것만 확인했다(검출 결과는 의미 없음).

## 후보 표

크기·라이선스는 아래 출처(HF API 메타데이터, GitHub 저장소 메타·README, 파일 HEAD 응답)에서 직접 확인했다. 가중치 라이선스가 저장소(코드) 라이선스와 같다고 보장되지는 않는다. 학습 데이터(COCO·Objects365 등)의 이용 조건까지 따진 것은 아니다 → 법무 확인 대상.

### A. 객체 검출(키보드·마우스·휴대폰, COCO 클래스 포함)

| 이름(측정 도구) | 출처 | 받을 파일 / 크기 | 라이선스 | 비고 |
|---|---|---|---|---|
| `rtdetr-r18` | huggingface.co/PekingU/rtdetr_r18vd_coco_o365 | model.safetensors 80.9MB (+설정 파일) | Apache-2.0 (HF 카드) | 가장 작은 RT-DETR |
| `rtdetr-r50` | huggingface.co/PekingU/rtdetr_r50vd | model.safetensors 172.2MB | Apache-2.0 (HF 카드) | 정확도 기준선 |
| `dfine-s` | huggingface.co/ustc-community/dfine-small-coco | model.safetensors 41.5MB | Apache-2.0 (HF 카드) | 가장 작음 |
| DETR-R50 | huggingface.co/facebook/detr-resnet-50 | model.safetensors 166.6MB | Apache-2.0 (HF 카드) | 느린 편, 참고용(도구 미등록) |
| YOLOX-s/m ONNX | github.com/Megvii-BaseDetection/YOLOX 릴리스 0.1.1rc0 | yolox_s.onnx 35.9MB / yolox_m.onnx 101.3MB | 코드 Apache-2.0 (GitHub 메타). **가중치 조건 미확인**(README에 라이선스 문구 없음) | 도구 미등록(ONNX 어댑터 필요) |
| RF-DETR | github.com/roboflow/rf-detr | 미확인 | 코드 Apache-2.0 (GitHub 메타). 일부 크기·가중치 별도 조건 가능성 **미확인** | 도구 미등록 |
| **Ultralytics YOLO11n/s** | github.com/ultralytics/assets 릴리스 v8.3.0 | yolo11n.pt 5.6MB / yolo11s.pt 19.3MB | **AGPL-3.0**(상업 이용은 별도 Enterprise 라이선스 — 저장소 README) | 속도 기준선용으로만. 서비스 채택은 라이선스 결정 필요. 도구 미등록 |
| YOLOv9 / YOLO-World | github.com/WongKinYiu/yolov9, AILab-CVC/YOLO-World | 미조사 | **GPL-3.0** (GitHub 메타) | 서비스 코드와 결합 시 제약 → 제외 권장 |

### B. open-vocab (태블릿·메모·스마트워치)

| 이름 | 출처 | 받을 파일 / 크기 | 라이선스 |
|---|---|---|---|
| `owlv2` | huggingface.co/google/owlv2-base-patch16-ensemble | model.safetensors 619.9MB | Apache-2.0 (HF 카드) |
| `gdino-t` | huggingface.co/IDEA-Research/grounding-dino-tiny | model.safetensors 689.4MB | Apache-2.0 (HF 카드) |
| (선택) grounding-dino-base | huggingface.co/IDEA-Research/grounding-dino-base | 933.4MB | Apache-2.0 (HF 카드) |
| YOLO-World | 위 표 | — | GPL-3.0 (제외 권장) |

### C. 손

| 이름 | 출처 | 받을 파일 / 크기 | 라이선스 |
|---|---|---|---|
| `mediapipe-hands` | storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task | 7.8MB (+ pip `mediapipe`) | 코드 Apache-2.0 (GitHub 메타). **모델 파일 자체 라이선스는 미확인** (모델 카드 미열람) |
| 대체: RTMPose/RTMW 손 | github.com/open-mmlab/mmpose | 미조사 | 코드 Apache-2.0 (GitHub 메타), 가중치 미확인 |

## 받을 위치

모두 `models/` (git 제외). 폴더 이름은 `tools/bench/adapters.py` 의 REGISTRY 와 같다:
`models/rtdetr_r18vd_coco_o365/`, `rtdetr_r50vd/`, `dfine-small-coco/`, `owlv2-base-patch16-ensemble/`, `grounding-dino-tiny/`, `hand_landmarker.task`.

## 설계 문서와의 불일치(보고만, 문서는 수정 안 함)

- 설계 §7-3은 "사전학습 YOLO 계열로 시작"을 예시하지만 대표 구현(Ultralytics)은 AGPL-3.0이다. §15-2의 주의와 같은 문제이며, 위 표의 RT-DETR·D-FINE(Apache-2.0)이 대안이다.
- 설계 §7-1은 손목·손바닥 중심(0, 9) 또는 손가락 끝 과반으로 영역 판정이라 하나, 현재 `HandDet` 은 점 하나(x, y)만 담는다. 측정 도구는 0·9의 중점을 쓴다. 랜드마크 전체를 쓰려면 `HandDet` 확장이 필요하다.
