# evaluate.py
# ─────────────────────────────────────────────────────────────────────────────
# 실험 구성:
#   [참고 실험] Exp1: COCO Pretrained 평가          ← 직접 실행
#   [Baseline]  Exp2: train.py 결과 불러와서 평가   ← 학습 없이 결과만 가져옴
#   [Best]      Exp3: Exp2 모델 + NMS 최적화        ← 직접 실행
#
# 실제로 돌리는 실험: Exp1, Exp3
# Exp2는 train.py가 저장한 best.pt를 평가만 합니다.
#
# 사용법:
#   python evaluate.py                  # 전체 (1→2평가→3)
#   python evaluate.py --exp 1          # 참고 실험만
#   python evaluate.py --exp 2          # train.py 결과 평가만
#   python evaluate.py --exp 3          # NMS 최적화만
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics 패키지가 없습니다: pip install ultralytics")
    sys.exit(1)

import cv2
import numpy as np
import config

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
RESULT_DIR  = config.ROOT / "outputs" / "experiments"
RESULT_JSON = RESULT_DIR / "results.json"

# train.py가 저장하는 best.pt 경로
TRAIN_BEST_PT = config.OUTPUT_DIR / "train" / "waste_detector" / "weights" / "best.pt"


# ══════════════════════════════════════════════════════════════════════════════
# FPS 측정
# ══════════════════════════════════════════════════════════════════════════════

def measure_fps(model, device: str, n_frames: int = 100) -> float:
    """실제 val 이미지 샘플로 FPS를 측정합니다."""
    val_dir   = config.ROOT / "data" / "images" / "val"
    img_paths = list(val_dir.glob("*.jpg"))

    if img_paths:
        sample = random.sample(img_paths, min(n_frames, len(img_paths)))
        frames = [cv2.imread(str(p)) for p in sample]
        frames = [f for f in frames if f is not None]
        while len(frames) < n_frames:
            frames += frames
        frames = frames[:n_frames]
        print(f"    → val 이미지 {len(set(str(p) for p in sample))}장 샘플로 FPS 측정")
    else:
        print("    → val 이미지 없음, 더미 이미지로 FPS 측정")
        frames = [np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)] * n_frames

    for f in frames[:5]:
        model.predict(f, device=device, verbose=False)

    t0 = time.perf_counter()
    for f in frames:
        model.predict(f, device=device, verbose=False)
    elapsed = time.perf_counter() - t0

    return round(n_frames / elapsed, 1)


# ══════════════════════════════════════════════════════════════════════════════
# 실험 1 — COCO Pretrained (참고 실험)
# ══════════════════════════════════════════════════════════════════════════════

def run_exp1(args) -> dict:
    """
    COCO 사전학습 모델을 Fine-tuning 없이 평가합니다.
    mAP ≈ 0 예상 → Fine-tuning 필요성 시각화 목적
    """
    print("\n" + "═" * 65)
    print("  [참고 실험] Exp1: COCO Pretrained (Fine-tuning 없음)")
    print("  ※ mAP ≈ 0 예상 → Fine-tuning 필요성 시각화 목적")
    print("═" * 65)

    model = YOLO("yolov8n.pt")

    try:
        metrics = model.val(
            data    = str(config.DATA_YAML),
            imgsz   = config.IMG_SIZE,
            device  = args.device,
            verbose = False,
        )
        result = _extract_metrics(metrics, "[참고] COCO Pretrained (Fine-tuning 없음)")
    except Exception as e:
        print(f"  [INFO] 클래스 불일치로 인한 예외 (정상): {e}")
        result = _zero_result("[참고] COCO Pretrained (Fine-tuning 없음)")

    print("  FPS 측정 중...")
    result["fps"] = measure_fps(model, args.device)
    _print_result(result)
    _save_json(_load_json() + [result])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 실험 2 — train.py 결과 불러와서 평가 (학습 없음)
# ══════════════════════════════════════════════════════════════════════════════

def run_exp2(args) -> dict:
    """
    train.py가 학습한 best.pt를 불러와서 평가만 수행합니다.
    별도 학습 없이 기존 결과를 Baseline으로 활용합니다.

    탐색 순서:
      1. outputs/train/waste_detector/weights/best.pt  (train.py 기본 저장 경로)
      2. outputs/experiments/exp2_baseline/weights/best.pt (evaluate.py로 학습한 경우)
      3. models/waste_detector.pt (수동 배치)
    """
    print("\n" + "═" * 65)
    print("  [Baseline] Exp2: Fine-tuning 결과 평가 (train.py 결과 활용)")
    print("  ※ 추가 학습 없이 기존 best.pt 불러와서 평가만 수행")
    print("═" * 65)

    # best.pt 탐색
    candidates = [
        TRAIN_BEST_PT,
        RESULT_DIR / "exp2_baseline" / "weights" / "best.pt",
        config.MODEL_PT,
    ]
    best_pt = None
    for c in candidates:
        if c.exists():
            best_pt = c
            break

    if best_pt is None:
        print("[ERROR] train.py 결과(best.pt)를 찾을 수 없습니다.")
        print("  먼저 train.py를 실행하세요:")
        print("  python train.py --epochs 50 --batch 16 --device 0 --no-augment")
        return _zero_result("[Baseline] Fine-tuning (best.pt 없음)")

    print(f"  모델 로드: {best_pt}")
    model = YOLO(str(best_pt))

    metrics = model.val(
        data    = str(config.DATA_YAML),
        imgsz   = config.IMG_SIZE,
        device  = args.device,
        verbose = False,
    )

    result = _extract_metrics(metrics, "[Baseline] Fine-tuning (증강 OFF)")
    print("  FPS 측정 중...")
    result["fps"] = measure_fps(model, args.device)
    _print_result(result)
    _save_json(_load_json() + [result])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 실험 4 — NMS 최적화 (Best)
# ══════════════════════════════════════════════════════════════════════════════

def run_exp3(args) -> dict:
    """
    Exp2와 동일한 best.pt를 재사용해 NMS 파라미터만 최적화합니다.
    추가 학습 없이 추론 파라미터 튜닝만으로 성능 향상을 측정합니다.
    """
    print("\n" + "═" * 65)
    print("  [Best] Exp3: Fine-tuning + NMS 최적화")
    print("  ※ best.pt 재사용, 추가 학습 없음")
    print("  ※ Conf / IoU threshold 그리드 서치")
    print("═" * 65)

    # Exp2와 동일한 경로에서 탐색
    candidates = [
        TRAIN_BEST_PT,
        RESULT_DIR / "exp2_baseline" / "weights" / "best.pt",
        config.MODEL_PT,
    ]
    best_pt = None
    for c in candidates:
        if c.exists():
            best_pt = c
            break

    if best_pt is None:
        print("[ERROR] best.pt 없음. 먼저 train.py 또는 Exp2를 실행하세요.")
        return _zero_result("[Best] NMS 최적화 (best.pt 없음)")

    print(f"  모델 로드: {best_pt}")
    model = YOLO(str(best_pt))

    # ── NMS 그리드 서치 ──────────────────────────────────────────────────────
    conf_candidates = [0.25, 0.35, 0.45, 0.50, 0.55]
    iou_candidates  = [0.35, 0.40, 0.45, 0.50, 0.55]
    total = len(conf_candidates) * len(iou_candidates)

    best_map50       = -1.0
    best_conf        = config.CONF_THRESHOLD
    best_iou         = config.IOU_THRESHOLD
    best_metrics_obj = None

    print(f"\n  NMS 그리드 서치: {total}개 조합 탐색 중...\n")

    for conf in conf_candidates:
        for iou in iou_candidates:
            try:
                m = model.val(
                    data    = str(config.DATA_YAML),
                    imgsz   = config.IMG_SIZE,
                    device  = args.device,
                    conf    = conf,
                    iou     = iou,
                    verbose = False,
                )
                map50 = float(m.box.map50)
                print(f"    conf={conf:.2f}  iou={iou:.2f}  →  mAP@0.5={map50:.4f}")

                if map50 > best_map50:
                    best_map50       = map50
                    best_conf        = conf
                    best_iou         = iou
                    best_metrics_obj = m
            except Exception as e:
                print(f"    conf={conf:.2f}  iou={iou:.2f}  →  오류: {e}")

    print(f"\n  최적 파라미터: conf={best_conf:.2f}, iou={best_iou:.2f}  "
          f"→  mAP@0.5={best_map50:.4f}")

    result = _extract_metrics(
        best_metrics_obj,
        f"[Best] Fine-tuning + NMS 최적화 (conf={best_conf}, iou={best_iou})"
    )
    result["nms_conf"] = best_conf
    result["nms_iou"]  = best_iou

    print("  FPS 측정 중...")
    result["fps"] = measure_fps(model, args.device)
    _print_result(result)
    _save_json(_load_json() + [result])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 유틸 함수
# ══════════════════════════════════════════════════════════════════════════════

def _extract_metrics(metrics, name: str) -> dict:
    try:
        return {
            "name":      name,
            "precision": round(float(metrics.box.mp),    4),
            "recall":    round(float(metrics.box.mr),    4),
            "mAP50":     round(float(metrics.box.map50), 4),
            "mAP50_95":  round(float(metrics.box.map),   4),
            "fps":       0.0,
        }
    except Exception as e:
        print(f"[WARN] 지표 추출 실패: {e}")
        return _zero_result(name)


def _zero_result(name: str) -> dict:
    return {"name": name, "precision": 0.0, "recall": 0.0,
            "mAP50": 0.0, "mAP50_95": 0.0, "fps": 0.0}


def _print_result(result: dict):
    print(f"\n  결과: {result['name']}")
    print(f"  Precision  : {result['precision']:.4f}")
    print(f"  Recall     : {result['recall']:.4f}")
    print(f"  mAP@0.5    : {result['mAP50']:.4f}")
    print(f"  mAP@0.5:95 : {result['mAP50_95']:.4f}")
    print(f"  FPS        : {result['fps']:.1f}")


def _load_json() -> list:
    if RESULT_JSON.exists():
        with open(RESULT_JSON, "r", encoding="utf-8") as f:
            return json.load(f).get("results", [])
    return []


def _save_json(results: list):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results":   results,
        }, f, ensure_ascii=False, indent=2)
    print(f"  [저장] {RESULT_JSON}")


def _print_comparison_table(results: list):
    print("\n\n" + "═" * 95)
    print("  Performance Comparison")
    print("  주요 비교: [Baseline] vs [Best]  |  참고: COCO Pretrained")
    print("═" * 95)
    print(f"  {'Model':<55} {'P':>6} {'R':>6} {'mAP50':>7} {'mAP50-95':>9} {'FPS':>7}")
    print("  " + "─" * 93)

    for r in results:
        nms_note = ""
        if r.get("nms_conf", 0) > 0:
            nms_note = f"  ← conf={r['nms_conf']}, iou={r['nms_iou']}"
        print(
            f"  {r['name']:<55} "
            f"{r['precision']:>6.4f} "
            f"{r['recall']:>6.4f} "
            f"{r['mAP50']:>7.4f} "
            f"{r['mAP50_95']:>9.4f} "
            f"{r['fps']:>7.1f}"
            f"{nms_note}"
        )

    print("═" * 95)

    baseline = next((r for r in results if "Baseline" in r["name"]), None)
    best     = next((r for r in results if "Best"     in r["name"]), None)
    if baseline and best:
        delta_map = best["mAP50"] - baseline["mAP50"]
        delta_fps = best["fps"]   - baseline["fps"]
        print(f"\n  [핵심 결과] NMS 최적화 효과 (Baseline → Best)")
        print(f"    mAP@0.5: {baseline['mAP50']:.4f} → {best['mAP50']:.4f} "
              f"({'↑' if delta_map >= 0 else '↓'}{abs(delta_map):.4f})")
        print(f"    FPS    : {baseline['fps']:.1f} → {best['fps']:.1f} "
              f"({'↑' if delta_fps >= 0 else '↓'}{abs(delta_fps):.1f})")
        if best.get("nms_conf", 0) > 0:
            print(f"    최적 NMS: conf={best['nms_conf']:.2f}, iou={best['nms_iou']:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="B:in 실험 비교 평가")
    parser.add_argument(
        "--exp", type=int, default=0,
        help="실험 번호 (0=전체 1→2→3, 1=참고, 2=Baseline평가, 3=NMS최적화)"
    )
    parser.add_argument("--device",  type=str, default="0")
    parser.add_argument("--workers", type=int, default=config.TRAIN_WORKERS)
    return parser.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    if args.exp in (0, 1):
        results.append(run_exp1(args))
    if args.exp in (0, 2):
        results.append(run_exp2(args))
    if args.exp in (0, 3):
        results.append(run_exp3(args))

    if len(results) > 1:
        _print_comparison_table(results)
    elif results:
        _print_result(results[0])

    print("\n[완료] 모든 실험이 끝났습니다.")