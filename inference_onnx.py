# inference_onnx.py
# ─────────────────────────────────────────────────────────────────────────────
# ONNX Runtime을 사용해 단일 이미지 또는 폴더 전체에 추론을 수행합니다.
# PyTorch / GPU 없이 CPU만으로 실행 가능합니다.
#
# 사용법:
#   python inference_onnx.py --source data/test_images/bottle.jpg
#   python inference_onnx.py --source data/test_images/ --save
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("[ERROR] onnxruntime 패키지가 없습니다:")
    print("  CPU 전용: pip install onnxruntime")
    print("  GPU 지원: pip install onnxruntime-gpu")
    sys.exit(1)

import config
import utils


# ══════════════════════════════════════════════════════════════════════════════
# ONNX Runtime 세션 관리자
# ══════════════════════════════════════════════════════════════════════════════

class ONNXDetector:
    """
    ONNX Runtime을 사용하는 YOLOv8 폐기물 탐지기.

    사용 예:
        detector = ONNXDetector("models/waste_detector.onnx")
        detections = detector.detect(image)
    """

    def __init__(
        self,
        onnx_path: str | Path = config.MODEL_ONNX,
        conf_threshold: float = config.CONF_THRESHOLD,
        iou_threshold:  float = config.IOU_THRESHOLD,
        providers: list | None = None,
    ):
        self.onnx_path      = Path(onnx_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold

        if not self.onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX 모델이 없습니다: {self.onnx_path}\n"
                "export_onnx.py를 먼저 실행하세요."
            )

        # Execution Provider 설정 (GPU 가능하면 CUDA, 아니면 CPU)
        if providers is None:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                print("[ONNXDetector] CUDA GPU 사용")
            else:
                providers = ["CPUExecutionProvider"]
                print("[ONNXDetector] CPU 모드")

        # 세션 옵션 설정
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_options,
            providers=providers,
        )

        # 입출력 이름 확인
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1, 3, H, W]

        # 입력 크기 추출 (동적 배치의 경우 config 값 사용)
        self.img_size = (
            self.input_shape[2]
            if isinstance(self.input_shape[2], int) and self.input_shape[2] > 0
            else config.IMG_SIZE
        )

        print(f"[ONNXDetector] 모델 로드 완료: {self.onnx_path}")
        print(f"  입력: {self.input_name} {self.input_shape}")
        print(f"  출력: {self.output_name}")

        # 워밍업 (첫 추론 지연 제거)
        self._warmup()

    def _warmup(self, n: int = 3):
        """더미 이미지로 모델을 워밍업합니다."""
        dummy = np.zeros((1, 3, self.img_size, self.img_size), dtype=np.float32)
        for _ in range(n):
            self.session.run([self.output_name], {self.input_name: dummy})

    def detect(
        self,
        image: np.ndarray,
        conf_threshold: float | None = None,
        iou_threshold:  float | None = None,
    ) -> tuple[list[dict], float]:
        """
        BGR 이미지에서 폐기물을 탐지합니다.

        Args:
            image:          BGR numpy 배열
            conf_threshold: Confidence 임계값 (None이면 self.conf_threshold 사용)
            iou_threshold:  IoU 임계값 (None이면 self.iou_threshold 사용)

        Returns:
            (detections, inference_time_ms)
            detections: [{"bbox", "confidence", "class_id", "label"}, ...]
        """
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou  = iou_threshold  if iou_threshold  is not None else self.iou_threshold

        orig_shape = image.shape[:2]  # (H, W)

        # 전처리: letterbox → 배치 텐서
        padded, ratio, (pad_w, pad_h) = utils.letterbox(
            image, new_shape=(self.img_size, self.img_size)
        )
        inp = utils.preprocess_image(padded, self.img_size)

        # 추론
        t0 = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: inp})
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000

        # 후처리: NMS
        detections = utils.non_max_suppression(outputs[0], conf, iou)

        # 좌표를 원본 이미지 크기로 변환
        detections = self._rescale_boxes(
            detections, orig_shape, self.img_size, ratio, pad_w, pad_h
        )

        return detections, inference_ms

    @staticmethod
    def _rescale_boxes(
        detections: list[dict],
        orig_shape: tuple,
        img_size: int,
        ratio: float,
        pad_w: float,
        pad_h: float,
    ) -> list[dict]:
        """
        letterbox 좌표 → 원본 이미지 좌표로 역변환합니다.
        """
        orig_h, orig_w = orig_shape
        rescaled = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            # 패딩 제거
            x1 -= pad_w;  x2 -= pad_w
            y1 -= pad_h;  y2 -= pad_h
            # 스케일 복원
            x1 /= ratio;  x2 /= ratio
            y1 /= ratio;  y2 /= ratio
            # 클리핑
            x1 = max(0.0, min(x1, float(orig_w)))
            y1 = max(0.0, min(y1, float(orig_h)))
            x2 = max(0.0, min(x2, float(orig_w)))
            y2 = max(0.0, min(y2, float(orig_h)))
            rescaled.append({**det, "bbox": [x1, y1, x2, y2]})
        return rescaled


# ══════════════════════════════════════════════════════════════════════════════
# 단일 이미지 추론
# ══════════════════════════════════════════════════════════════════════════════

def run_image(
    detector: ONNXDetector,
    image_path: Path,
    save: bool = False,
    show: bool = True,
):
    """단일 이미지 파일을 추론하고 결과를 출력합니다."""
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[WARN] 이미지 로드 실패: {image_path}")
        return

    detections, ms = detector.detect(image)

    print(f"\n{'─' * 50}")
    print(f"파일: {image_path.name}")
    print(f"추론 시간: {ms:.1f} ms  ({1000/ms:.1f} FPS)")
    print(f"탐지 수:   {len(detections)}")

    for i, det in enumerate(detections, 1):
        label_ko = config.CLASS_NAMES_KO.get(det["label"], det["label"])
        guide    = utils.get_disposal_guide(det["label"])
        print(f"\n  [{i}] {label_ko}  (신뢰도: {det['confidence']:.1%})")
        print(f"      bbox: {[f'{v:.0f}' for v in det['bbox']]}")
        print(f"      분리배출: {guide}")

    # 시각화
    vis = utils.draw_detections(image, detections)

    if save:
        out_path = config.OUTPUT_DIR / f"result_{image_path.name}"
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), vis)
        print(f"\n  저장: {out_path}")

    if show:
        cv2.imshow("B:in Detection", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
# 폴더 배치 추론
# ══════════════════════════════════════════════════════════════════════════════

def run_folder(detector: ONNXDetector, folder: Path, save: bool = True):
    """폴더 내 모든 이미지를 배치 추론합니다."""
    exts  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [p for p in folder.iterdir() if p.suffix.lower() in exts]
    print(f"[Batch] {len(paths)}개 이미지 처리 중...")

    total_ms = 0.0
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        detections, ms = detector.detect(image)
        total_ms += ms

        if save:
            vis = utils.draw_detections(image, detections)
            out = config.OUTPUT_DIR / f"result_{path.name}"
            config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), vis)

    if paths:
        avg_ms = total_ms / len(paths)
        print(f"[Batch] 완료. 평균 추론 시간: {avg_ms:.1f} ms ({1000/avg_ms:.1f} FPS)")


# ══════════════════════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="ONNX Runtime 폐기물 탐지 추론")
    parser.add_argument("--model",  type=Path,  default=config.MODEL_ONNX, help="ONNX 모델 경로")
    parser.add_argument("--source", type=Path,  required=True,             help="이미지 파일 또는 폴더 경로")
    parser.add_argument("--conf",   type=float, default=config.CONF_THRESHOLD, help="Confidence 임계값")
    parser.add_argument("--iou",    type=float, default=config.IOU_THRESHOLD,  help="NMS IoU 임계값")
    parser.add_argument("--save",   action="store_true", default=True,    help="결과 이미지 저장")
    parser.add_argument("--no-show",action="store_true", default=False,   help="화면 표시 비활성화")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    detector = ONNXDetector(
        onnx_path      = args.model,
        conf_threshold = args.conf,
        iou_threshold  = args.iou,
    )

    source = args.source
    if source.is_dir():
        run_folder(detector, source, save=args.save)
    elif source.is_file():
        run_image(detector, source, save=args.save, show=not args.no_show)
    else:
        print(f"[ERROR] 경로가 존재하지 않습니다: {source}")
        sys.exit(1)