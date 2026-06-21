# utils.py
# ─────────────────────────────────────────────────────────────────────────────
# 이미지 전처리, 바운딩 박스 시각화, 색상 매핑, 분리배출 가이드 반환을 담당합니다.
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
from collections import deque
from typing import Optional

import config


# ══════════════════════════════════════════════════════════════════════════════
# 1. 이미지 전처리
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_image(image: np.ndarray, img_size: int = config.IMG_SIZE) -> np.ndarray:
    """
    ONNX Runtime 추론에 필요한 형태로 이미지를 전처리합니다.

    Steps:
        BGR → RGB → Resize → Normalize(0~1) → HWC → CHW → Batch 차원 추가

    Args:
        image:    BGR numpy 배열 (OpenCV 기본 포맷)
        img_size: 모델 입력 크기 (기본 640)

    Returns:
        shape (1, 3, img_size, img_size), float32, 0~1 범위
    """
    # BGR → RGB 변환
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 리사이즈 (letterbox 없이 단순 resize — ONNX 추론 데모용)
    resized = cv2.resize(rgb, (img_size, img_size))

    # 정규화: 0~255 → 0.0~1.0
    normalized = resized.astype(np.float32) / 255.0

    # HWC → CHW
    chw = np.transpose(normalized, (2, 0, 1))

    # 배치 차원 추가: (3, H, W) → (1, 3, H, W)
    batched = np.expand_dims(chw, axis=0)

    return batched


def letterbox(
    image: np.ndarray,
    new_shape: tuple = (640, 640),
    color: tuple = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple]:
    """
    비율을 유지하면서 이미지를 리사이즈하고 패딩을 추가합니다.
    (YOLOv8 공식 전처리 방식)

    Returns:
        (padded_image, scale_ratio, (pad_w, pad_h))
    """
    h, w = image.shape[:2]
    target_h, target_w = new_shape

    # 스케일 비율 계산 (작은 쪽 기준)
    ratio = min(target_h / h, target_w / w)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))

    # 리사이즈
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 패딩 계산 (중앙 정렬)
    pad_w = (target_w - new_w) / 2
    pad_h = (target_h - new_h) / 2
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right  = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color,
    )

    return padded, ratio, (pad_w, pad_h)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 후처리 (NMS)
# ══════════════════════════════════════════════════════════════════════════════

def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    YOLO 출력 형식 (cx, cy, w, h) → (x1, y1, x2, y2) 변환
    """
    out = boxes.copy()
    out[..., 0] = boxes[..., 0] - boxes[..., 2] / 2  # x1
    out[..., 1] = boxes[..., 1] - boxes[..., 3] / 2  # y1
    out[..., 2] = boxes[..., 0] + boxes[..., 2] / 2  # x2
    out[..., 3] = boxes[..., 1] + boxes[..., 3] / 2  # y2
    return out


def non_max_suppression(
    predictions: np.ndarray,
    conf_threshold: float = config.CONF_THRESHOLD,
    iou_threshold:  float = config.IOU_THRESHOLD,
) -> list[dict]:
    """
    ONNX 모델 출력에 NMS를 적용합니다.

    Args:
        predictions: shape (1, num_classes+4, num_anchors) — YOLOv8 raw output
        conf_threshold: Confidence 임계값
        iou_threshold:  IoU 임계값

    Returns:
        탐지 결과 리스트 [{"bbox", "confidence", "class_id", "label"}, ...]
    """
    # (1, 8400, 4+nc) 형태로 변환
    pred = predictions[0]            # (4+nc, 8400)
    pred = pred.T                    # (8400, 4+nc)

    boxes_xywh = pred[:, :4]         # 박스 좌표
    scores_all  = pred[:, 4:]        # 클래스별 점수

    # 가장 높은 클래스 점수를 confidence로 사용
    class_ids   = np.argmax(scores_all, axis=1)
    confidences = scores_all[np.arange(len(scores_all)), class_ids]

    # Confidence 임계값 필터링
    mask = confidences >= conf_threshold
    if not np.any(mask):
        return []

    boxes_xywh  = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids   = class_ids[mask]

    # xywh → xyxy 변환
    boxes_xyxy = xywh2xyxy(boxes_xywh)

    # OpenCV NMS 적용
    indices = cv2.dnn.NMSBoxes(
        bboxes     = boxes_xyxy.tolist(),
        scores     = confidences.tolist(),
        score_threshold = conf_threshold,
        nms_threshold   = iou_threshold,
    )

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            cls_id = int(class_ids[i])
            label  = (config.CLASS_NAMES[cls_id]
                      if cls_id < len(config.CLASS_NAMES)
                      else f"class_{cls_id}")
            results.append({
                "bbox":       [float(v) for v in boxes_xyxy[i]],
                "confidence": float(confidences[i]),
                "class_id":  cls_id,
                "label":     label,
            })

    # Confidence 내림차순 정렬
    results.sort(key=lambda x: x["confidence"], reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3. 시각화
# ══════════════════════════════════════════════════════════════════════════════

def get_class_color(label: str) -> tuple:
    """클래스 이름에 따른 BGR 색상을 반환합니다."""
    return config.CLASS_COLORS.get(label, config.DEFAULT_COLOR)


def draw_detections(
    image: np.ndarray,
    detections: list[dict],
    show_conf: bool = True,
) -> np.ndarray:
    """
    이미지 위에 바운딩 박스와 레이블을 그립니다.

    Args:
        image:      BGR numpy 배열
        detections: non_max_suppression() 반환값
        show_conf:  Confidence 점수 표시 여부

    Returns:
        바운딩 박스가 그려진 BGR 이미지
    """
    h, w = image.shape[:2]
    output = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]

        # 좌표를 이미지 크기에 맞게 클리핑
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w - 1))
        y2 = max(0, min(int(y2), h - 1))

        label   = det["label"]
        conf    = det["confidence"]
        color   = get_class_color(label)
        label_ko = config.CLASS_NAMES_KO.get(label, label)

        # 표시 텍스트 구성
        text = f"{label_ko}"
        if show_conf:
            text += f" {conf:.0%}"

        # 바운딩 박스
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        # 레이블 배경
        (tw, th), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        label_y = max(y1 - 4, th + baseline)
        cv2.rectangle(
            output,
            (x1, label_y - th - baseline),
            (x1 + tw + 4, label_y + baseline),
            color, -1
        )

        # 레이블 텍스트
        cv2.putText(
            output, text,
            (x1 + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 2, cv2.LINE_AA,
        )

    return output


def scale_boxes(
    boxes: list,
    orig_shape: tuple,
    input_size: int = config.IMG_SIZE,
) -> list:
    """
    모델 입력 해상도 기준 좌표를 원본 이미지 해상도로 변환합니다.

    Args:
        boxes:      [{"bbox": [x1,y1,x2,y2], ...}, ...]
        orig_shape: (orig_h, orig_w) — 원본 이미지 크기
        input_size: 모델 입력 크기

    Returns:
        좌표가 원본 기준으로 스케일된 boxes
    """
    orig_h, orig_w = orig_shape[:2]
    scale_x = orig_w / input_size
    scale_y = orig_h / input_size

    scaled = []
    for det in boxes:
        x1, y1, x2, y2 = det["bbox"]
        scaled.append({
            **det,
            "bbox": [
                x1 * scale_x, y1 * scale_y,
                x2 * scale_x, y2 * scale_y,
            ],
        })
    return scaled


# ══════════════════════════════════════════════════════════════════════════════
# 4. 분리배출 가이드
# ══════════════════════════════════════════════════════════════════════════════

def get_disposal_guide(label: str, sub_label: Optional[str] = None) -> str:
    """
    클래스 레이블에 해당하는 분리배출 안내 문자열을 반환합니다.

    Args:
        label:     1차 분류 결과 (e.g. "pet_bottle")
        sub_label: 2차 분류 결과 (e.g. "transparent_plastic") — 있으면 세부 안내

    Returns:
        분리배출 안내 문자열
    """
    if sub_label and sub_label in config.DISPOSAL_GUIDE:
        return config.DISPOSAL_GUIDE[sub_label]
    return config.DISPOSAL_GUIDE.get(label, "해당 재질 수거함에 배출하세요.")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Video Smoothing (Moving Average)
# ══════════════════════════════════════════════════════════════════════════════

class BBoxSmoother:
    """
    바운딩 박스 좌표에 Moving Average를 적용해 깜빡임을 줄입니다.

    각 클래스별로 최근 N 프레임의 좌표 평균을 사용합니다.
    """

    def __init__(self, window: int = config.SMOOTH_WINDOW):
        self.window = window
        # 클래스별 좌표 이력: {"label": deque([bbox, ...], maxlen=window)}
        self._history: dict[str, deque] = {}

    def smooth(self, detections: list[dict]) -> list[dict]:
        """
        탐지 결과 리스트에 Moving Average를 적용합니다.

        Args:
            detections: [{"bbox", "confidence", "class_id", "label"}, ...]

        Returns:
            좌표가 평활화된 탐지 결과 리스트
        """
        smoothed = []
        seen_labels: set[str] = set()

        for det in detections:
            label = det["label"]
            seen_labels.add(label)

            if label not in self._history:
                self._history[label] = deque(maxlen=self.window)

            self._history[label].append(det["bbox"])

            # 이력 평균 계산
            avg_bbox = np.mean(list(self._history[label]), axis=0).tolist()
            smoothed.append({**det, "bbox": avg_bbox})

        # 이번 프레임에 없는 클래스의 이력 삭제 (다음 등장 시 이전 값 오염 방지)
        for label in list(self._history.keys()):
            if label not in seen_labels:
                del self._history[label]

        return smoothed