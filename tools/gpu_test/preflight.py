"""GPU 합격 시험 앞의 짧은 점검(1~2분): 모델을 직접 올려
1) fp16 과 fp32 결과 비교(D-FINE·OWLv2 점수 차이, 검출 라벨 일치) - fp16 정확도 하락 여부
2) 단계별 묶음 속도 곡선(D-FINE·OWLv2·RTMW 를 1·4·8·16장 등으로 묶을 때 장당 ms) - 용량 추정의 가장 큰 미지수
3) GPU·CUDA·ONNX Runtime provider 확인
을 표로 낸다. GPU 가 없으면 CPU 로 흐름만 확인한다(속도 값은 의미 없음). 서버를 띄우기 전에 돈다(GPU 메모리를 나눠 쓰지 않게).

  python tools/gpu_test/preflight.py --models-dir models --images <make_images 폴더> [--out preflight.json] [--quick]
종료 코드: 0 정상, 1 fp16 결과가 달라 일부 모델을 fp32 로 둘 것(마지막 줄 FP16_SKIP=...), 3 실행 오류(설치·모델 문제).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

SCORE_DIFF_LIMIT = 0.05  # fp16 과 fp32 의 최고 점수 차이 허용(초과면 경고)


def load(models_dir: str, fp16: bool, simulate_cpu_fp16: bool = False):
    from pair_vision.models.batch_real import RealBatchBackend
    from pair_vision.rules.config import load_rules

    rules = load_rules(ROOT / "configs" / "rules.default.yaml").models
    import os

    force = {"1": True, "0": False}.get(os.environ.get("VISION_DEVICE_PREPROCESS", "").strip())
    det = RealBatchBackend(models_dir, replace(rules, fp16=fp16), verify=False, warmup=False, device_preprocess=force)
    det.load()
    if fp16 and simulate_cpu_fp16 and det._device == "cpu":  # 흐름 확인용: CPU 에서 반정밀도 경로를 억지로 탄다(느림)
        det._dfine_amp, det._amp_device = True, "cpu"
        det._owl_half, det._dtype = True, det._torch.float16
        det._owl.half()
    return det


def images(folder: Path):
    import numpy as np
    from PIL import Image

    phone = [np.asarray(Image.open(p).convert("RGB")) for p in sorted((folder / "phone").glob("*.jpg"))]
    webcam = [np.asarray(Image.open(p).convert("RGB")) for p in sorted((folder / "webcam").glob("*.jpg"))]
    if not phone or not webcam:
        raise SystemExit(f"사진 없음: {folder} (make_images.py 먼저)")
    return phone, webcam


def detections(det, rgb, which: str, threshold: float = 0.3) -> list[tuple[int, float, list[float]]]:
    """모델 최종 검출(모든 클래스·질의, 점수 >= threshold): (클래스, 점수, 상자). DETR 류는 질의 순서가 정밀도에 따라
    바뀌므로(상위 k 선택) 질의끼리가 아니라 최종 검출끼리 비교한다."""
    size = [(rgb.shape[0], rgb.shape[1])]
    torch = det._torch
    if which == "dfine":
        out = det._dfine_forward([det.coco_pixels(rgb, det._device)])
        out.logits, out.pred_boxes = out.logits.float(), out.pred_boxes.float()
        res = det._dfine_proc.post_process_object_detection(out, threshold=threshold, target_sizes=size)[0]
    else:
        with torch.inference_mode():
            out = det._owl(input_ids=det._owl_ids, attention_mask=det._owl_mask,
                           pixel_values=det._stack([det.owl_pixels(rgb, det._device)], det._dtype))
        out.logits, out.pred_boxes = out.logits.float(), out.pred_boxes.float()
        side = max(size[0])
        post = getattr(det._owl_proc, "post_process_grounded_object_detection", None) or det._owl_proc.post_process_object_detection
        res = post(out, threshold=threshold, target_sizes=[(side, side)])[0]
    return [(int(lb), float(sc), bx.tolist()) for sc, lb, bx in zip(res["scores"], res["labels"], res["boxes"], strict=True)]


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compare(fast_dets, ref_dets, margin: float = 0.1) -> tuple[int, float]:
    """(짝 없는 검출 수, 짝지은 검출의 최대 점수 차). 기준값 근처(점수 0.3+margin 미만)에서 한쪽에만 있는 것은 세지 않는다."""
    unmatched, worst, used = 0, 0.0, set()
    for lb, sc, bx in ref_dets:
        best = max(((iou(bx, b2), j) for j, (l2, _, b2) in enumerate(fast_dets) if l2 == lb and j not in used), default=(0, -1))
        if best[0] >= 0.5:
            used.add(best[1])
            worst = max(worst, abs(sc - fast_dets[best[1]][1]))
        elif sc >= 0.3 + margin:
            unmatched += 1
    unmatched += sum(1 for j, (_, sc, _) in enumerate(fast_dets) if j not in used and sc >= 0.3 + margin)
    return unmatched, worst


def timed(det, fn, repeat: int) -> float:
    torch = det._torch
    fn()
    if det._device == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(repeat):
        fn()
    if det._device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / repeat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--quick", action="store_true", help="CPU 흐름 확인용: 묶음 1·2장만, 반복 1번")
    ap.add_argument("--simulate-cpu-fp16", action="store_true", help="CPU 에서 반정밀도 비교 경로를 억지로 실행(흐름 확인용)")
    args = ap.parse_args()
    report: dict = {}
    try:
        import onnxruntime as ort
        import torch

        report["env"] = {"torch": torch.__version__, "cuda": torch.cuda.is_available(),
                         "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                         "onnxruntime": ort.__version__, "ort_providers": ort.get_available_providers()}
        print("환경:", report["env"], flush=True)
        phone, webcam = images(Path(args.images))
        t = time.perf_counter()
        fast = load(args.models_dir, fp16=True, simulate_cpu_fp16=args.simulate_cpu_fp16)
        report["precision"] = fast.precision
        from pair_vision.models.preprocess import gpu_decode_available

        report["device_preprocess"] = fast.device_preprocess  # 전부 GPU 경로(풀기·전처리·자르기·레터박스)
        report["gpu_jpeg_decode"] = gpu_decode_available() and fast._device == "cuda"
        report["rtmw_gpu"] = fast._rtmw.gpu
        print(f"모델 적재 {time.perf_counter() - t:.1f}초 (정밀도 {fast.precision}, RTMW GPU {fast._rtmw.gpu}, "
              f"전처리 전부 {fast._device if fast.device_preprocess else 'CPU'}, JPEG GPU 풀기 {report['gpu_jpeg_decode']})", flush=True)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"점검 실패(설치·모델 문제): {exc!r}", file=sys.stderr)
        return 3

    # 1) fp16 vs fp32 (모델별 최종 검출 비교) -> 결과가 다른 모델은 fp32 로 두라고 알린다(fp16_skip)
    fp16_skip: list[str] = []
    if fast.precision != {"dfine": "fp32", "owlv2": "fp32"}:
        ref = load(args.models_dir, fp16=False)
        per: dict = {}
        for which, name in (("dfine", "dfine"), ("owl", "owlv2")):
            un, worst, n = 0, 0.0, 0
            for rgb in phone[:6]:
                a = detections(fast, rgb, which)
                b = detections(ref, rgb, which)
                u, w = compare(a, b)
                un, worst, n = un + u, max(worst, w), n + len(b)
            per[name] = {"fp32_detections": n, "unmatched": un, "max_score_diff": round(worst, 4)}
            if un > 0 or worst > SCORE_DIFF_LIMIT:
                fp16_skip.append(name)
        del ref
        if fast._device == "cuda":
            fast._torch.cuda.empty_cache()
        report["fp16"] = {"precision": fast.precision, "per_model": per, "limit": SCORE_DIFF_LIMIT, "fp16_skip": fp16_skip}
        for name, v in per.items():
            print(f"fp16 vs fp32 {name}: fp32 검출 {v['fp32_detections']}개, 짝 없음 {v['unmatched']}, 최대 점수 차 "
                  f"{v['max_score_diff']} -> {'fp32 로 둠' if name in fp16_skip else 'fp16 사용'}", flush=True)
    else:
        report["fp16"] = {"skipped": "GPU 아님 -> fp32", "fp16_skip": []}
        print("fp16 비교 건너뜀: GPU 가 아니어서 fp32 로 돈다", flush=True)

    # 2) 묶음 속도 곡선
    rgb, bgr = phone[0], phone[0][:, :, ::-1].copy()
    wbgr = webcam[0][:, :, ::-1].copy()
    sizes = [(rgb.shape[0], rgb.shape[1])]
    rep = 1 if args.quick else 5
    plan = {
        "dfine": ((1, 2) if args.quick else (1, 4, 8, 16)),
        "owlv2": ((1, 2) if args.quick else (1, 2, 4)),
        "rtmw": ((1, 2) if args.quick else (1, 8, 16, 32)),
    }
    curve: dict = {}
    crop = fast._rtmw.crop(bgr, fast._rtmw.full_frame(rgb.shape[1], rgb.shape[0])[0])
    for n in plan["dfine"]:
        curve.setdefault("dfine", {})[n] = timed(
            fast, lambda n=n: fast.coco_batch([fast.coco_pixels(rgb, fast._device) for _ in range(n)], sizes * n), rep) / n
    for n in plan["owlv2"]:
        curve.setdefault("owlv2", {})[n] = timed(
            fast, lambda n=n: fast.owl_batch([fast.owl_pixels(rgb, fast._device) for _ in range(n)], sizes * n), rep) / n
    for n in plan["rtmw"]:
        curve.setdefault("rtmw", {})[n] = timed(fast, lambda n=n: fast._rtmw.pose_batch([crop] * n), rep) / n
    curve["person"] = {1: timed(fast, lambda: fast._rtmw.person_boxes(wbgr), rep)}
    report["batch_curve_ms_per_image"] = {k: {str(n): round(v * 1000, 2) for n, v in d.items()} for k, d in curve.items()}
    print("\n| 단계 | 묶음 크기별 장당 ms (전처리 포함) | 배치 이득(최대 묶음/1장) |\n|---|---|---|")
    for k, d in curve.items():
        ns = sorted(d)
        gain = d[ns[-1]] / d[ns[0]] if len(ns) > 1 else 1.0
        print(f"| {k} | " + ", ".join(f"{n}장 {d[n] * 1000:.1f}" for n in ns) + f" | {gain:.2f} |")
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    if fp16_skip:
        print(f"주의: {fp16_skip} 는 fp16 결과가 fp32 와 달라 서버를 이 모델만 fp32 로 띄웁니다(VISION_FP16_SKIP). 보고서에 기록됩니다.")
    print("FP16_SKIP=" + ",".join(fp16_skip))
    return 1 if fp16_skip else 0


if __name__ == "__main__":
    raise SystemExit(main())
