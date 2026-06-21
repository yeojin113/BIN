# train.py
# ─────────────────────────────────────────────────────────────────────────────
# YOLOv8n 기반 폐기물 탐지 모델 학습 스크립트
#
# 사용법:
#   python train.py
#   python train.py --epochs 50 --batch 8
#
# 데이터 준비:
#   data/dataset.yaml 파일과 data/images/, data/labels/ 폴더가 필요합니다.
#   Roboflow에서 YOLOv8 형식으로 다운로드 후 data/ 에 배치하세요.
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import os
import shutil
import sys
from pathlib import Path

# ── 패키지 임포트 ─────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics 패키지가 없습니다. 아래 명령어로 설치하세요:")
    print("  pip install ultralytics")
    sys.exit(1)

try:
    import albumentations as A
    import yaml
    import cv2
    import numpy as np
    _ALB_AVAILABLE = True
except ImportError:
    _ALB_AVAILABLE = False
    print("[WARNING] albumentations 미설치 → 기본 증강만 사용합니다.")
    print("  pip install albumentations")

import config


# ══════════════════════════════════════════════════════════════════════════════
# Albumentations 증강 파이프라인
# ══════════════════════════════════════════════════════════════════════════════

def build_augmentation_pipeline() -> "A.Compose":
    """
    YOLOv8 학습용 Albumentations 증강 파이프라인을 구성합니다.

    각 변환은 bbox_params를 통해 바운딩 박스도 함께 변환됩니다.
    """
    return A.Compose(
        [
            # ── 기하학적 변환 ──────────────────────────────────────────────
            A.HorizontalFlip(p=0.5),                    # 좌우 반전
            A.VerticalFlip(p=0.1),                      # 상하 반전 (드물게)
            A.RandomRotate90(p=0.2),                    # 90도 단위 회전
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                p=0.4,
            ),

            # ── 색상·밝기 변환 ────────────────────────────────────────────
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.3,
            ),
            A.CLAHE(clip_limit=2.0, p=0.2),             # 국소 대비 향상
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2), # 가우시안 노이즈

            # ── 블러·선명도 ───────────────────────────────────────────────
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),      # 모션 블러
                A.GaussianBlur(blur_limit=5, p=1.0),    # 가우시안 블러
                A.MedianBlur(blur_limit=5, p=1.0),      # 미디언 블러
            ], p=0.2),

            # ── 차폐(Occlusion) 시뮬레이션 ───────────────────────────────
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                min_holes=1,
                fill_value=0,
                p=0.3,
            ),

            # ── 이미지 압축 시뮬레이션 ────────────────────────────────────
            A.ImageCompression(quality_lower=75, quality_upper=100, p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="yolo",           # YOLO 형식: (cx, cy, w, h) 정규화
            label_fields=["labels"],
            min_visibility=0.3,      # 증강 후 30% 미만 가시 박스 제거
        ),
    )


def apply_augmentation_to_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    num_augments: int = 3,
):
    """
    데이터셋 전체에 Albumentations 증강을 적용하고 저장합니다.

    Args:
        images_dir:        원본 이미지 폴더
        labels_dir:        원본 라벨 폴더 (.txt, YOLO 형식)
        output_images_dir: 증강 이미지 저장 폴더
        output_labels_dir: 증강 라벨 저장 폴더
        num_augments:      이미지 1장당 생성할 증강 이미지 수
    """
    if not _ALB_AVAILABLE:
        print("[SKIP] albumentations 미설치로 증강을 건너뜁니다.")
        return

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    transform = build_augmentation_pipeline()

    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    print(f"[Augment] 총 {len(image_paths)}장 이미지에 ×{num_augments} 증강 적용 중...")

    for img_path in image_paths:
        # 이미지 로드
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 대응 라벨 파일 로드
        label_path = labels_dir / (img_path.stem + ".txt")
        bboxes, labels = [], []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:])
                        bboxes.append([cx, cy, bw, bh])
                        labels.append(cls_id)

        # 원본 복사
        out_img_path = output_images_dir / img_path.name
        out_lbl_path = output_labels_dir / label_path.name
        shutil.copy(img_path, out_img_path)
        if label_path.exists():
            shutil.copy(label_path, out_lbl_path)

        # 증강 이미지 생성
        for idx in range(num_augments):
            try:
                augmented = transform(image=image, bboxes=bboxes, labels=labels)
            except Exception as e:
                print(f"  [WARN] {img_path.name} 증강 실패: {e}")
                continue

            aug_img   = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
            aug_boxes = augmented["bboxes"]
            aug_lbls  = augmented["labels"]

            # 저장 파일명
            stem = f"{img_path.stem}_aug{idx}"
            cv2.imwrite(str(output_images_dir / f"{stem}.jpg"), aug_img)

            with open(output_labels_dir / f"{stem}.txt", "w") as f:
                for cls_id, (cx, cy, bw, bh) in zip(aug_lbls, aug_boxes):
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    print(f"[Augment] 완료. 저장 위치: {output_images_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# 학습 메인 함수
# ══════════════════════════════════════════════════════════════════════════════

def train(args):
    """YOLOv8 모델을 학습하고 best.pt를 models/ 에 복사합니다."""

    # ── (선택) Albumentations 증강 적용 ──────────────────────────────────────
    if args.augment and _ALB_AVAILABLE:
        print("\n[Step 1] Albumentations 데이터 증강 적용 중...")
        aug_images = config.ROOT / "data" / "augmented" / "images" / "train"
        aug_labels = config.ROOT / "data" / "augmented" / "labels" / "train"
        apply_augmentation_to_dataset(
            images_dir        = config.ROOT / "data" / "images" / "train",
            labels_dir        = config.ROOT / "data" / "labels" / "train",
            output_images_dir = aug_images,
            output_labels_dir = aug_labels,
            num_augments      = args.num_augments,
        )
        # 증강된 데이터를 사용하도록 yaml 경로 업데이트
        data_yaml = config.ROOT / "data" / "dataset_aug.yaml"
        _write_augmented_yaml(data_yaml)
    else:
        print("\n[Step 1] 증강 없이 원본 데이터셋 사용")
        data_yaml = config.DATA_YAML

    # ── YOLOv8 모델 로드 ──────────────────────────────────────────────────────
    print(f"\n[Step 2] 사전학습 모델 로드: {args.pretrained}")
    model = YOLO(args.pretrained)

    # ── 학습 실행 ─────────────────────────────────────────────────────────────
    print(f"\n[Step 3] 학습 시작")
    print(f"  data:   {data_yaml}")
    print(f"  epochs: {args.epochs}")
    print(f"  batch:  {args.batch}")
    print(f"  imgsz:  {args.imgsz}")
    print(f"  device: {args.device}\n")

    results = model.train(
        data      = str(data_yaml),
        epochs    = args.epochs,
        batch     = args.batch,
        imgsz     = args.imgsz,
        device    = args.device,
        workers   = args.workers,
        project   = str(config.OUTPUT_DIR / "train"),
        name      = "waste_detector",
        exist_ok  = True,
        patience  = 20,          # Early stopping: 20 epoch 개선 없으면 중단
        save      = True,
        plots     = True,        # 학습 곡선, Confusion Matrix 자동 저장
        val       = True,        # 매 epoch 검증 수행
        # 빌트인 Mosaic, MixUp 증강
        mosaic    = 1.0,
        mixup     = 0.1,
    )

    # ── 모델 저장 ─────────────────────────────────────────────────────────────
    best_pt = config.OUTPUT_DIR / "train" / "waste_detector" / "weights" / "best.pt"
    if best_pt.exists():
        config.MODEL_PT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_pt, config.MODEL_PT)
        print(f"\n[Step 4] 최적 가중치 저장 완료: {config.MODEL_PT}")
    else:
        print(f"\n[WARN] best.pt를 찾을 수 없습니다: {best_pt}")

    # ── 성능 평가 출력 ────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  성능 평가 결과 (검증 세트 기준)")
    print("═" * 60)
    _print_metrics(model)

    return results


def _print_metrics(model):
    """학습 완료 후 검증 세트 성능 지표를 출력합니다."""
    try:
        metrics = model.val()
        print(f"  Precision  (P)  : {metrics.box.mp:.4f}")
        print(f"  Recall     (R)  : {metrics.box.mr:.4f}")
        print(f"  mAP@0.5        : {metrics.box.map50:.4f}")
        print(f"  mAP@0.5:0.95   : {metrics.box.map:.4f}")
    except Exception as e:
        print(f"  [WARN] 지표 출력 실패: {e}")
    print("═" * 60)


def _write_augmented_yaml(output_path: Path):
    """증강 데이터를 사용하는 dataset yaml 파일을 생성합니다."""
    content = {
        "path":  str(config.ROOT / "data" / "augmented"),
        "train": "images/train",
        "val":   str(config.ROOT / "data" / "images" / "val"),
        "nc":    len(config.CLASS_NAMES),
        "names": config.CLASS_NAMES,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        import yaml
        yaml.dump(content, f, allow_unicode=True, default_flow_style=False)
    print(f"[YAML] 증강 데이터셋 yaml 생성: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="B:in YOLOv8 학습 스크립트")
    parser.add_argument("--epochs",       type=int,   default=config.TRAIN_EPOCHS,  help="학습 에폭 수")
    parser.add_argument("--batch",        type=int,   default=config.TRAIN_BATCH,   help="배치 크기")
    parser.add_argument("--imgsz",        type=int,   default=config.IMG_SIZE,      help="입력 이미지 크기")
    parser.add_argument("--device",       type=str,   default="0",                  help="GPU 번호 또는 'cpu'")
    parser.add_argument("--workers",      type=int,   default=config.TRAIN_WORKERS, help="데이터 로더 워커 수")
    parser.add_argument("--pretrained",   type=str,   default=config.PRETRAINED,    help="사전학습 모델")
    parser.add_argument("--augment",      action="store_true",  default=True,        help="Albumentations 증강 적용")
    parser.add_argument("--no-augment",   action="store_false", dest="augment",      help="증강 비활성화")
    parser.add_argument("--num-augments", type=int,   default=3,                    help="이미지당 증강 횟수")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)