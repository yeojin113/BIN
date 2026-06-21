# app.py
# ─────────────────────────────────────────────────────────────────────────────
# B:in Streamlit 웹 앱  (실시간 카메라 + 사진 촬영 + 이미지 업로드)
#
# 실행 방법:
#   streamlit run app.py
#
# 입력 방식:
# 입력 방식:
#   - 📸 사진 촬영 / 이미지 업로드 : 파일 업로드 (폰에서는 촬영 선택 가능)
#   - 📹 실시간 카메라 : 브라우저 카메라 실시간 추론 (localhost/https 필요)
# ─────────────────────────────────────────────────────────────────────────────

import csv
import time
import threading
from datetime import datetime
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

import config
import utils

# ── 실시간 카메라(webrtc)용 — 미설치 시 실시간 모드만 비활성화 ────────────────
try:
    import av
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    _USE_WEBRTC = True
except Exception:
    _USE_WEBRTC = False

# ── ONNX 또는 PyTorch 탐지기 선택 (ONNX 우선) ────────────────────────────────
try:
    from inference_onnx import ONNXDetector
    _USE_ONNX = True
except Exception:
    _USE_ONNX = False

try:
    from ultralytics import YOLO
    _USE_PT = True
except Exception:
    _USE_PT = False


# ══════════════════════════════════════════════════════════════════════════════
# 페이지 설정
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="B:in | 스마트 분리수거 도우미",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# 클래스별 색상 팔레트
# ══════════════════════════════════════════════════════════════════════════════

DET_COLORS = {
    "pet_bottle": "#3b82f6",
    "vinyl":      "#a855f7",
    "can":        "#10b981",
    "paper":      "#f59e0b",
}
_PALETTE = ["#6366f1", "#ec4899", "#14b8a6", "#f97316", "#06b6d4"]


def _color_hex(label: str, class_id: int = 0) -> str:
    return DET_COLORS.get(label, _PALETTE[class_id % len(_PALETTE)])


def _hex_to_bgr(h: str):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ══════════════════════════════════════════════════════════════════════════════
# 한글 폰트 로더
# ══════════════════════════════════════════════════════════════════════════════

_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]


@st.cache_resource(show_spinner=False)
def _load_font(size: int = 22):
    for path in _FONT_CANDIDATES:
        if not Path(path).exists():
            continue
        for idx in range(0, 8):
            try:
                f = ImageFont.truetype(path, size, index=idx)
                if f.getlength("가") > 0:
                    return f
            except Exception:
                continue
    return ImageFont.load_default()


# ── 전역 CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;600;700&family=Space+Grotesk:wght@500;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
.bin-logo { font-family: 'Space Grotesk', sans-serif; font-size: 2.6rem; font-weight: 700;
  letter-spacing: -0.03em; color: #111827; margin-bottom: 0; }
.bin-logo span { color: #16a34a; }
.status-row { display:flex; gap:8px; flex-wrap:wrap; margin:6px 0 2px; }
.pill { font-family:'Space Grotesk', sans-serif; display:inline-flex; align-items:center; gap:6px;
  background:#f3f4f6; color:#374151; border:1px solid #e5e7eb; border-radius:999px;
  padding:4px 12px; font-size:0.78rem; font-weight:600; }
.pill .dot { width:7px; height:7px; border-radius:50%; background:#16a34a;
             box-shadow:0 0 0 3px rgba(22,163,74,.18); }
.card-green  { background:#f0fdf4; border:2px solid #22c55e; border-radius:14px; padding:1.2rem 1.4rem; margin-bottom:0.8rem; }
.card-blue   { background:#eff6ff; border:2px solid #60a5fa; border-radius:14px; padding:1.2rem 1.4rem; margin-bottom:0.8rem; }
.card-orange { background:#fff7ed; border:2px solid #fb923c; border-radius:14px; padding:1.2rem 1.4rem; margin-bottom:0.8rem; }
.card-title { font-size:0.82rem; font-weight:600; color:#6b7280; margin-bottom:0.3rem; }
.card-value { font-family:'Space Grotesk', sans-serif; font-size:1.7rem; font-weight:700; color:#111827; }
.card-sub   { font-size:0.9rem; color:#374151; margin-top:0.4rem; line-height:1.6; }
@keyframes fillbar { from { width:0; } to { width:var(--w); } }
.conf-track { background:#e5e7eb; border-radius:999px; height:11px; overflow:hidden; margin-top:10px; }
.conf-fill  { height:100%; border-radius:999px; background:linear-gradient(90deg,#34d399,#16a34a);
              animation: fillbar .8s cubic-bezier(.22,1,.36,1) forwards; }
.badge { display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.75rem; font-weight:600; }
.badge-green  { background:#dcfce7; color:#15803d; }
.badge-blue   { background:#dbeafe; color:#1d4ed8; }
.badge-orange { background:#ffedd5; color:#c2410c; }
.badge-live   { background:#fee2e2; color:#b91c1c; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }
.stat-row { display:flex; justify-content:space-between; align-items:center;
            padding:6px 0; border-bottom:1px solid #e5e7eb; font-size:0.9rem; }
.stat-row .dot { width:9px; height:9px; border-radius:3px; margin-right:6px; }
.stat-cnt { font-weight:700; color:#16a34a; }
.eco-msg { background:linear-gradient(90deg,#dcfce7,#f0fdf4); border-radius:12px;
           padding:0.9rem 1.2rem; margin:0.6rem 0; font-size:0.95rem; color:#15803d;
           font-weight:600; border:1px solid #bbf7d0; }
.warn-box { background:#fff7ed; border:1px solid #fdba74; border-radius:10px;
            padding:0.7rem 1rem; margin-top:0.5rem; font-size:0.85rem; color:#9a3412; }
div[data-testid="stMetricValue"] { font-family:'Space Grotesk', sans-serif; font-size:1.5rem !important; }
.stProgress > div > div { background:#16a34a !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 세션 상태 초기화
# ══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "detector":   None,
        "smoother":   utils.BBoxSmoother(),
        "counts":     defaultdict(int),
        "total":      0,
        "conf_sum":   0.0,
        "infer_ms":   [],
        "history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def track_latency(ms: float):
    st.session_state.infer_ms.append(ms)
    if len(st.session_state.infer_ms) > 60:
        st.session_state.infer_ms = st.session_state.infer_ms[-60:]


# ══════════════════════════════════════════════════════════════════════════════
# 피드백 저장 (human-in-the-loop)
# ══════════════════════════════════════════════════════════════════════════════

FEEDBACK_DIR     = config.ROOT / "feedback"
FEEDBACK_IMG_DIR = FEEDBACK_DIR / "images"
FEEDBACK_LOG     = FEEDBACK_DIR / "feedback_log.csv"
_FEEDBACK_FIELDS = ["timestamp", "image_file", "predicted", "correct",
                    "predicted_conf", "backend"]


def save_feedback(bgr_image, predicted_label, predicted_conf, correct_label, backend):
    FEEDBACK_IMG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_name = f"{ts}_{predicted_label}_to_{correct_label}.jpg"
    img_path = FEEDBACK_IMG_DIR / img_name
    try:
        cv2.imwrite(str(img_path), bgr_image)
    except Exception as e:
        return None, f"이미지 저장 실패: {e}"
    new_file = not FEEDBACK_LOG.exists()
    try:
        with open(FEEDBACK_LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_FEEDBACK_FIELDS)
            if new_file:
                w.writeheader()
            w.writerow({
                "timestamp": ts, "image_file": img_name,
                "predicted": predicted_label, "correct": correct_label,
                "predicted_conf": f"{predicted_conf:.4f}", "backend": backend,
            })
    except Exception as e:
        return None, f"로그 저장 실패: {e}"
    return img_path, None


def load_feedback_log():
    if not FEEDBACK_LOG.exists():
        return []
    try:
        with open(FEEDBACK_LOG, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
# 탐지기 로더 (캐시 — 싱글톤)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="모델 로딩 중...")
def load_detector(conf_thr: float, iou_thr: float):
    if _USE_ONNX and config.MODEL_ONNX.exists():
        return ONNXDetector(config.MODEL_ONNX, conf_thr, iou_thr), "ONNX"
    if _USE_PT and config.MODEL_PT.exists():
        return _PyTorchDetector(config.MODEL_PT, conf_thr, iou_thr), "PyTorch"
    return _DemoDetector(conf_thr), "Demo"


class _PyTorchDetector:
    def __init__(self, pt_path, conf, iou):
        self.model = YOLO(str(pt_path))
        self.conf = conf; self.iou = iou

    def detect(self, image, conf_threshold=None, iou_threshold=None):
        conf = conf_threshold or self.conf
        iou  = iou_threshold  or self.iou
        t0 = time.perf_counter()
        results = self.model.predict(image, conf=conf, iou=iou, verbose=False)
        ms = (time.perf_counter() - t0) * 1000
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                detections.append({
                    "label": config.CLASS_NAMES[cls_id] if cls_id < len(config.CLASS_NAMES) else f"cls_{cls_id}",
                    "class_id": cls_id,
                    "confidence": float(box.conf[0]),
                    "bbox": list(map(float, box.xyxy[0].tolist())),
                })
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections, ms


class _DemoDetector:
    def __init__(self, conf):
        self.conf = conf

    def detect(self, image, conf_threshold=None, iou_threshold=None):
        import random
        h, w = image.shape[:2]
        if random.random() < 0.45:
            return [], 12.0
        cls_id = random.randint(0, len(config.CLASS_NAMES) - 1)
        conf = round(random.uniform(0.60, 0.97), 3)
        bw = random.randint(w // 5, w // 3); bh = random.randint(h // 5, h // 3)
        x1 = random.randint(0, w - bw); y1 = random.randint(0, h - bh)
        return [{
            "label": config.CLASS_NAMES[cls_id], "class_id": cls_id,
            "confidence": conf,
            "bbox": [float(x1), float(y1), float(x1+bw), float(y1+bh)],
        }], 12.0


# ══════════════════════════════════════════════════════════════════════════════
# 박스 드로잉
# ══════════════════════════════════════════════════════════════════════════════

def draw_boxes(bgr: np.ndarray, detections: list, show_conf: bool = True) -> np.ndarray:
    vis = bgr.copy()
    H, W = vis.shape[:2]
    box_meta = []
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        hexc  = _color_hex(det["label"], det.get("class_id", 0))
        color = _hex_to_bgr(hexc)
        overlay = vis.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
        L = max(16, int(min(x2 - x1, y2 - y1) * 0.22)); t = 3
        for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1),
                               (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            cv2.line(vis, (cx, cy), (cx + dx * L, cy), color, t, cv2.LINE_AA)
            cv2.line(vis, (cx, cy), (cx, cy + dy * L), color, t, cv2.LINE_AA)
        ko  = config.CLASS_NAMES_KO.get(det["label"], det["label"])
        txt = f"{ko}  {det['confidence']*100:.0f}%" if show_conf else ko
        box_meta.append((x1, y1, txt, hexc))
    if not box_meta:
        return vis
    rgb  = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    pil  = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = _load_font(22)
    for x1, y1, txt, hexc in box_meta:
        fill = _hex_to_rgb(hexc)
        l, t_, r_, b_ = draw.textbbox((0, 0), txt, font=font)
        tw, th = r_ - l, b_ - t_; pad = 6
        ly = y1 - th - 2 * pad
        if ly < 0:
            ly = y1 + 2
        draw.rectangle([x1, ly, x1 + tw + 2 * pad, ly + th + 2 * pad], fill=fill)
        draw.text((x1 + pad, ly + pad - t_), txt, font=font, fill=(255, 255, 255))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# 클래스별 상세 정보 (분리배출/효과/마크/예외품목) — 출처: 환경부·서울시
# ══════════════════════════════════════════════════════════════════════════════

MARK_DIR = config.ROOT / "assets" / "marks"

DISPOSAL_INFO = {
    "pet_bottle": ["내용물을 비우고 물로 헹굽니다", "라벨(비닐)을 제거합니다",
                   "압착 후 뚜껑을 닫아 배출합니다", "투명 페트병은 별도 분리배출합니다"],
    "vinyl":      ["이물질을 제거한 후 배출합니다", "운송장·테이프를 제거합니다",
                   "부피를 줄여 배출합니다", "오염이 심하면 일반쓰레기로 배출합니다"],
    "can":        ["내용물을 비우고 헹굽니다", "찌그러뜨려 부피를 줄입니다",
                   "플라스틱 뚜껑은 분리합니다", "담배꽁초 등 이물질을 넣지 않습니다"],
    "paper":      ["비닐·테이프를 제거합니다", "물기에 젖지 않게 배출합니다",
                   "종이팩은 일반 종이와 구분합니다", "코팅 종이는 일반쓰레기로 배출합니다"],
}

# 비닐처럼 보여도 일반쓰레기인 예외 품목
EXCEPTION_INFO = {
    "pet_bottle": "색깔 페트병(세제·화학제품 용기), 펌프 뚜껑(금속 스프링), 칫솔·볼펜, 빨대는 일반쓰레기로 배출하세요.",
    "vinyl":      "오염된 랩·비닐, 돗자리, 고무장갑, 식탁보, 장판, 현수막, 은박비닐은 일반쓰레기로 배출하세요.",
    "can":        "부탄가스·살충제 캔은 구멍을 뚫어 잔가스를 제거 후 배출하세요. 페인트통 등 이물질 캔은 주의가 필요합니다.",
    "paper":      "영수증·전표, 코팅지, 기름 묻은 종이(치킨박스), 벽지는 일반쓰레기로 배출하세요.",
}

# 감성 환경 메시지
ECO_MESSAGE = {
    "pet_bottle": "🌱 지구지킴이님 덕분에 페트병이 새 옷으로 다시 태어나요!",
    "vinyl":      "💚 깨끗한 비닐 분리배출, 지구가 고마워해요!",
    "can":        "♻️ 캔 하나가 모여 소중한 자원이 됩니다. 감사해요!",
    "paper":      "🌳 종이를 아껴주셔서 나무 한 그루가 숨 쉴 수 있어요!",
}

IMPACT_INFO = {
    "pet_bottle": ["재생 PET 원료 생산 가능", "새 플라스틱 사용량 감소"],
    "vinyl":      ["재생 플라스틱 생산 가능", "매립·소각 폐기물 감소"],
    "can":        ["금속 채굴량 감소", "재생 시 에너지 사용량 절감"],
    "paper":      ["나무 사용량 감소", "물과 에너지 절약"],
}

MARKS = {
    "pet_bottle": ["pet_yellow.png", "pet_blue.png"],
    "vinyl":      ["pet_vinyl.png", "pp_vinyl.png", "hdpe_vinyl.png",
                   "ldpe_vinyl.png", "ps_vinyl.png", "other_vinyl.png"],
    "can":        ["can_alu.png", "can.png"],
    "paper":      ["paper.png", "paperpack.png", "sterilepack.png"],
}

MARK_DESC = {
    "pet_yellow.png": "투명 페트병(전용)", "pet_blue.png": "PET 플라스틱",
    "pet_vinyl.png": "비닐 PET", "pp_vinyl.png": "비닐 PP",
    "hdpe_vinyl.png": "비닐 HDPE", "ldpe_vinyl.png": "비닐 LDPE",
    "ps_vinyl.png": "비닐 PS", "other_vinyl.png": "비닐 OTHER",
    "can_alu.png": "알루미늄 캔", "can.png": "철 캔",
    "paper.png": "종이류", "paperpack.png": "종이팩", "sterilepack.png": "멸균팩",
}

# 재활용 과정 (출처: 분리의 정석 / 내 손안의 분리배출)
#   summary: 한 줄 설명, steps: 단계 흐름, result: 최종 재생 제품
RECYCLE_PROCESS = {
    "pet_bottle": {
        "summary": "분쇄·살균·세척·건조·성형 과정을 거쳐 식품용기나 장섬유 등 고급제품으로 재활용됩니다.",
        "steps": ["무색 페트병", "선별장 (이물질 제거)",
                  "재활용업체 (분쇄·살균·세척·건조·용융·성형)"],
        "result": "식품용기 · 고급 섬유(의류)",
    },
    "vinyl": {
        "summary": "파이프·배수로 등 건축자재로 물질재활용하거나, 고형연료제품(SRF)으로 재활용됩니다.",
        "steps": ["선별장 (이물질 제거)", "비닐 (압축운송)",
                  "재활용업체 (파쇄·분쇄·용융·성형 / SRF)"],
        "result": "파이프 · 배수관로 · 고형연료(SRF)",
    },
    "can": {
        "summary": "철캔·알루미늄캔 등으로 선별하여 제철소로 운반하고, 용융 과정을 거쳐 각각의 금속제품으로 재활용됩니다.",
        "steps": ["고철 선별장 (철캔·알루미늄캔·스테인리스 등)", "압축 운반",
                  "제철소 (용융)"],
        "result": "철판 · 알루미늄판 등",
    },
    "paper": {
        "summary": "해리·정선·농축·고해 과정을 거쳐 펄프와 혼합하여 골판지·신문지·화장지 등으로 다시 생산됩니다.",
        "steps": ["집하장 (선별)", "제지업", "해리·정선·농축·고해"],
        "result": "골판지 · 신문지 · 화장지 등",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 결과 렌더링
# ══════════════════════════════════════════════════════════════════════════════

def update_stats(detections: list):
    for det in detections:
        st.session_state.total    += 1
        st.session_state.conf_sum += det["confidence"]
        st.session_state.counts[det["label"]] += 1


def render_result_panel(detections, fps, result_ph, guide_ph, dev, show_guide):
    """탐지 결과 카드. 개발자 모드면 신뢰도·클래스명까지 표시."""
    if not detections:
        result_ph.markdown(
            '<div class="card-blue"><div class="card-title">탐지 결과</div>'
            '<div class="card-sub">폐기물을 촬영하거나 이미지를 업로드하세요.<br><br>'
            '📌 <strong>잘 안 잡힐 때 시도해보세요</strong><br>'
            '• 물체를 화면 가운데에<br>'
            '• 배경을 단순하게<br>'
            '• 최대한 가까이 촬영<br><br>'
            '• 지원하지 않는 품목(유리·스티로폼 등)은 위의 <strong>📚 분리배출 가이드</strong> 탭을 확인하세요'
            '</div></div>',
            unsafe_allow_html=True,
        )
        guide_ph.empty()
        return

    count = len(detections)
    cards_html = (f'<div class="card-title" style="margin-bottom:0.6rem;font-size:0.9rem;">'
                  f'감지된 폐기물 · 총 {count}개</div>')
    for i, det in enumerate(detections, start=1):
        label = det["label"]; label_ko = config.CLASS_NAMES_KO.get(label, label)
        conf = det["confidence"]; clr = _color_hex(label, det.get("class_id", 0))
        if dev:
            cards_html += (
                f'<div class="card-green" style="border-color:{clr};">'
                f'<div class="card-title">#{i}</div>'
                f'<div class="card-value" style="color:{clr};">{label_ko}</div>'
                f'<div class="card-sub">'
                f'{"✅ 높은 확신" if conf >= 0.7 else "⚠️ 확인 필요 — 다시 찍어보세요" if conf < 0.4 else "🔍 보통 확신"}'
                f'</div>'
                f'<div class="card-sub">신뢰도 <strong>{conf:.1%}</strong> &nbsp;|&nbsp; '
                f'클래스 <strong>{label}</strong></div>'
                f'<div class="conf-track"><div class="conf-fill" style="--w:{conf*100:.0f}%;'
                f'background:linear-gradient(90deg,{clr}99,{clr});"></div></div></div>'
            )
        else:
            cards_html += (
                f'<div class="card-green" style="border-color:{clr};">'
                f'<div class="card-value" style="color:{clr};">{label_ko}</div></div>'
            )
    result_ph.markdown(cards_html, unsafe_allow_html=True)

    # 분리배출 가이드 + 감성 메시지 + 예외품목 (클래스별 1회)
    if show_guide:
        guide_html = ""
        seen = set()
        for det in detections:
            label = det["label"]
            if label in seen:
                continue
            seen.add(label)
            label_ko = config.CLASS_NAMES_KO.get(label, label)
            clr = _color_hex(label, det.get("class_id", 0))
            steps = DISPOSAL_INFO.get(label, [])
            steps_html = "".join(f"<li>{s}</li>" for s in steps)
            exception = EXCEPTION_INFO.get(label, "")
            eco = ECO_MESSAGE.get(label, "")
            guide_html += (
                f'<div class="card-green" style="border-color:{clr};">'
                f'<div class="card-title" style="color:{clr};">✅ {label_ko} 분리배출 방법</div>'
                f'<ul style="font-size:0.88rem;color:#374151;line-height:1.7;'
                f'padding-left:1.1rem;margin:0.4rem 0;">{steps_html}</ul>'
                f'<div class="warn-box">⚠️ {exception}</div>'
                f'<div class="eco-msg">{eco}</div>'
                f'</div>'
            )
        guide_ph.markdown(guide_html, unsafe_allow_html=True)
    else:
        guide_ph.empty()


def render_marks_for_detections(detections):
    """탐지된 클래스의 분리배출 마크 이미지를 결과 아래에 표시."""
    if not detections:
        return
    seen = []
    for det in detections:
        if det["label"] not in seen:
            seen.append(det["label"])

    st.markdown("#### 🔖 이 폐기물의 분리배출 마크")
    for label in seen:
        label_ko = config.CLASS_NAMES_KO.get(label, label)
        marks = MARKS.get(label, [])
        marks = [m for m in marks if (MARK_DIR / m).exists()]
        if not marks:
            st.caption(f"{label_ko}: 마크 이미지 없음 (assets/marks/ 확인)")
            continue
        st.markdown(f"**{label_ko}**")
        cols = st.columns(min(len(marks), 6))
        for idx, mf in enumerate(marks):
            with cols[idx % len(cols)]:
                st.image(str(MARK_DIR / mf), use_container_width=True)
                st.caption(MARK_DESC.get(mf, ""))


def render_recycle_process(detections):
    """탐지된 클래스의 재활용 과정을 카드로 바로 표시. (출처: 분리의 정석)"""
    if not detections:
        return
    seen = []
    for det in detections:
        if det["label"] not in seen:
            seen.append(det["label"])

    for label in seen:
        proc = RECYCLE_PROCESS.get(label)
        if not proc:
            continue
        label_ko = config.CLASS_NAMES_KO.get(label, label)
        clr = _color_hex(label)

        # 단계 흐름 (화살표로 연결)
        flow = ""
        for i, step in enumerate(proc["steps"]):
            flow += (
                f'<span style="display:inline-block;background:#f0fdf4;'
                f'border:1.5px solid {clr};border-radius:10px;padding:6px 12px;'
                f'margin:3px;font-size:0.85rem;color:#15803d;font-weight:600;">'
                f'{i+1}. {step}</span>'
            )
            flow += '<span style="color:#9ca3af;margin:0 2px;">→</span>'
        flow += (
            f'<span style="display:inline-block;background:{clr};'
            f'border-radius:10px;padding:6px 14px;margin:3px;'
            f'font-size:0.88rem;color:#fff;font-weight:700;">'
            f'✨ {proc["result"]}</span>'
        )

        # 제목 + 요약 + 흐름 + 출처를 하나의 카드로 항상 표시
        st.markdown(
            f'<div style="background:#fff;border:1.5px solid {clr};border-radius:14px;'
            f'padding:1rem 1.3rem;margin:0.6rem 0;">'
            f'<div style="font-size:1.05rem;font-weight:700;color:{clr};'
            f'margin-bottom:0.4rem;">♻️ {label_ko}은(는) 이렇게 재활용돼요</div>'
            f'<div style="font-size:0.88rem;color:#374151;margin-bottom:0.6rem;'
            f'line-height:1.6;">{proc["summary"]}</div>'
            f'<div style="line-height:2.4;">{flow}</div>'
            f'<div style="font-size:0.78rem;color:#9ca3af;margin-top:0.5rem;">'
            f'출처: 분리의 정석 (내 손안의 분리배출)</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_feedback_widget(backend):
    if "last_detections" not in st.session_state:
        return
    dets = st.session_state["last_detections"]
    if not dets:
        return
    st.markdown("---")
    with st.expander("🛠️ 결과가 틀렸나요? 올바른 분류를 알려주세요"):
        # ── 1) 어느 탐지 결과가 틀렸는지 선택 ──────────────────────────────
        # 객체가 여러 개일 때, 각 박스를 구분해서 정정할 수 있도록 함
        det_options = []
        for i, d in enumerate(dets):
            ko = config.CLASS_NAMES_KO.get(d["label"], d["label"])
            det_options.append(f"#{i+1} {ko} ({d['confidence']:.0%})")

        st.caption("어떤 항목이 잘못 분류되었나요?")
        target_idx = st.radio(
            "정정할 항목",
            range(len(dets)),
            format_func=lambda i: det_options[i],
            key="fb_target",
        )
        target = dets[target_idx]
        pred_label = target["label"]
        pred_ko = config.CLASS_NAMES_KO.get(pred_label, pred_label)

        # ── 2) 그 항목의 올바른 분류 선택 ─────────────────────────────────
        st.caption(f"선택한 항목의 현재 예측: **{pred_ko}**")
        options_ko = [config.CLASS_NAMES_KO.get(c, c) for c in config.CLASS_NAMES]
        choice_ko = st.radio("올바른 분류는?", options_ko, horizontal=True, key="fb_choice")
        ko_to_label = {config.CLASS_NAMES_KO.get(c, c): c for c in config.CLASS_NAMES}
        correct_label = ko_to_label.get(choice_ko, choice_ko)

        # ── 3) 저장 ───────────────────────────────────────────────────────
        if st.button("📩 피드백 보내기", key="fb_submit"):
            if correct_label == pred_label:
                st.warning("예측과 동일한 분류를 선택했습니다. 다른 분류를 골라주세요.")
            else:
                path, err = save_feedback(
                    bgr_image=st.session_state["last_image"],
                    predicted_label=pred_label, predicted_conf=target["confidence"],
                    correct_label=correct_label, backend=backend,
                )
                if err:
                    st.error(f"저장 실패: {err}")
                else:
                    st.success("피드백이 기록되었습니다. 모델 개선에 활용됩니다. 감사합니다!")


# ══════════════════════════════════════════════════════════════════════════════
# 헤더
# ══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="bin-logo">B<span>:</span>in</div>', unsafe_allow_html=True)
st.caption("실시간 AI 분리수거 도우미 · YOLOv8 기반 폐기물 탐지 시스템")


# ══════════════════════════════════════════════════════════════════════════════
# 사이드바
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ 설정")

    # 입력 방식: 업로드 + 사진 촬영(camera_input) + 실시간 카메라
    mode_options = ["🖼️ 이미지 업로드", "📷 사진 촬영"]
    if _USE_WEBRTC:
        mode_options = mode_options + ["📹 실시간 카메라"]

    mode = st.radio("입력 방식", mode_options, index=0,
                    help="업로드는 폰에서 촬영/갤러리 선택이 가능합니다. '사진 촬영'과 '실시간 카메라'는 localhost/https 접속에서만 동작합니다.")

    if not _USE_WEBRTC:
        st.caption("⚠️ 실시간 카메라는 streamlit-webrtc 설치 시 활성화됩니다.")
    st.caption("📌 실시간 카메라는 브라우저 보안 정책상 "
               "`localhost` 또는 `https` 접속에서만 작동합니다. "
               "원격 IP 접속 시에는 사진 촬영/업로드를 사용하세요.")

    st.markdown("---")
    conf_thr = st.slider("Confidence Threshold", 0.10, 1.00, config.CONF_THRESHOLD, 0.05,
                         help="이 값 이상의 탐지만 표시합니다")
    st.caption("💡 권장값: **0.25** (precision/recall 균형 최적 구간)")
    iou_thr  = st.slider("IoU Threshold (NMS)", 0.10, 1.00, config.IOU_THRESHOLD, 0.05,
                         help="NMS에서 겹침 제거 기준")
    st.caption("💡 권장값: **0.50** (중복 박스 제거 안정 구간)")
    show_conf  = st.checkbox("Confidence 표시", value=True)
    show_guide = st.checkbox("분리배출 가이드 표시", value=True)

    detector, backend = load_detector(conf_thr, iou_thr)
    backend_color = {"ONNX": "badge-green", "PyTorch": "badge-blue", "Demo": "badge-orange"}
    st.markdown(f'<span class="badge {backend_color.get(backend,"badge-orange")}">'
                f'백엔드: {backend}</span>', unsafe_allow_html=True)
    if backend == "Demo":
        st.caption("⚠️ 모델 미연결 상태 — 탐지 결과는 임의값입니다.")

    st.markdown("---")
    st.markdown("## 📊 탐지 통계")
    total = st.session_state.total
    avg_conf = (st.session_state.conf_sum / total * 100) if total > 0 else 0
    c1, c2 = st.columns(2)
    c1.metric("총 탐지", f"{total}건")
    c2.metric("평균 신뢰도", f"{avg_conf:.1f}%" if total > 0 else "-")

    st.markdown("---")
    st.markdown("## 🕐 최근 탐지 기록")
    history = st.session_state.get("history", [])
    if history:
        for h in reversed(history):
            st.caption(f"{h['time']} — {', '.join(h['labels'])}")
    else:
        st.caption("아직 탐지 기록이 없습니다.")

    if st.session_state.counts:
        for lbl, cnt in sorted(st.session_state.counts.items(), key=lambda x: -x[1]):
            ko = config.CLASS_NAMES_KO.get(lbl, lbl); clr = _color_hex(lbl)
            st.markdown(f'<div class="stat-row"><span>'
                        f'<span class="dot" style="background:{clr};display:inline-block;"></span>{ko}</span>'
                        f'<span class="stat-cnt">{cnt}건</span></div>', unsafe_allow_html=True)

    if st.button("🗑️ 통계 초기화", use_container_width=True):
        st.session_state.counts = defaultdict(int)
        st.session_state.total = 0; st.session_state.conf_sum = 0.0
        st.session_state.infer_ms = []
        st.rerun()

    st.markdown("---")
    dev = st.checkbox("🔧 개발자 모드", value=False,
                      help="신뢰도·클래스명 등 상세 정보와 오분류 피드백을 확인합니다.")
    st.session_state["dev_mode"] = dev
    if dev:
        fb = load_feedback_log()
        st.markdown(f"**수집된 피드백: {len(fb)}건**")
        if fb:
            import pandas as pd
            df_fb = pd.DataFrame(fb)
            df_fb["예측→정답"] = df_fb["predicted"] + " → " + df_fb["correct"]
            st.caption("오분류 패턴")
            for pat, cnt in df_fb["예측→정답"].value_counts().items():
                st.markdown(f'<div class="stat-row"><span>{pat}</span>'
                            f'<span class="stat-cnt">{cnt}건</span></div>', unsafe_allow_html=True)
            st.caption("최근 피드백 기록")
            st.dataframe(df_fb[["timestamp","predicted","correct","predicted_conf"]].tail(20).iloc[::-1],
                         use_container_width=True, hide_index=True)
            csv_bytes = df_fb.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ 피드백 로그 CSV", csv_bytes,
                               file_name="feedback_log.csv", mime="text/csv",
                               use_container_width=True)
        else:
            st.info("아직 수집된 피드백이 없습니다.")


# ── 시스템 상태 pill ──────────────────────────────────────────────────────────
try:
    model_name = config.MODEL_ONNX.name if (_USE_ONNX and config.MODEL_ONNX.exists()) \
        else (config.MODEL_PT.name if (_USE_PT and config.MODEL_PT.exists()) else "demo")
except Exception:
    model_name = "demo"

st.markdown(
    f'<div class="status-row">'
    f'<span class="pill"><span class="dot"></span>SYSTEM ONLINE</span>'
    f'<span class="pill">BACKEND · {backend}</span>'
    f'<span class="pill">MODEL · {model_name}</span>'
    f'<span class="pill">CONF · {conf_thr:.2f}</span>'
    f'<span class="pill">IoU · {iou_thr:.2f}</span></div>',
    unsafe_allow_html=True,
)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# 실시간 카메라용 VideoProcessor
#   - 캐시된 detector를 직접 들고 사용 (스레드 안전)
#   - 무거운 추론을 매 프레임 하지 않도록 N프레임마다 1회만 추론
# ══════════════════════════════════════════════════════════════════════════════

if _USE_WEBRTC:
    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            # 캐시된 싱글톤 detector를 가져와 스레드에서 직접 사용
            self.detector, _ = load_detector(conf_thr, iou_thr)
            self.conf = conf_thr
            self.iou = iou_thr
            self.show_conf = show_conf
            self.frame_count = 0
            self.last_detections = []
            self.skip = 3   # 3프레임마다 1회 추론

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1
            # 무거운 추론은 skip 간격으로만, 나머지 프레임은 직전 결과 박스 재사용
            if self.frame_count % self.skip == 0:
                try:
                    dets, _ = self.detector.detect(img, self.conf, self.iou)
                    self.last_detections = dets
                except Exception:
                    self.last_detections = []
            img = draw_boxes(img, self.last_detections, self.show_conf)
            return av.VideoFrame.from_ndarray(img, format="bgr24")


def run_single_image(bgr, frame_ph, result_ph, guide_ph):
    with st.spinner("🔍 분석 중..."):
        detections, ms = detector.detect(bgr, conf_thr, iou_thr)
        detections = st.session_state.smoother.smooth(detections)
        update_stats(detections)

        # 히스토리 저장 (최근 5개)
        if detections:
            from datetime import datetime
            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M"),
                "labels": [config.CLASS_NAMES_KO.get(d["label"], d["label"]) for d in detections],
            })
            if len(st.session_state.history) > 5:
                st.session_state.history = st.session_state.history[-5:]

        track_latency(ms)
    vis = draw_boxes(bgr, detections, show_conf=show_conf)
    frame_ph.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)
    render_result_panel(detections, 0.0, result_ph, guide_ph, dev, show_guide)
    st.session_state["last_image"] = bgr
    st.session_state["last_detections"] = detections
    if dev:
        st.caption(f"추론 시간: {ms:.1f} ms ({1000/ms:.0f} FPS 상당)")
    # 탐지된 클래스 마크 이미지 표시
    render_marks_for_detections(detections)
    # 재활용 과정 (접이식)
    render_recycle_process(detections)

# ══════════════════════════════════════════════════════════════════════════════
# 메인 레이아웃 — 탭 구조
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2 = st.tabs(["♻️ AI 분류", "📚 분리배출 가이드"])

# ── TAB 1: AI 분류 ────────────────────────────────────────────────────────────
with tab1:
    col_cam, col_res = st.columns([3, 2], gap="large")
    with col_cam:
        st.markdown("#### 📷 입력 영상")
        frame_ph = st.empty()
    with col_res:
        st.markdown("#### 🔍 분류 결과")
        result_ph = st.empty()
        st.markdown("#### 📋 분리배출 안내")
        guide_ph = st.empty()

    # ── 모드별 분기 ──────────────────────────────────────────────────────────
    if mode == "📹 실시간 카메라":
        frame_ph.empty()
        st.info("카메라를 폐기물에 비추면 실시간으로 분류합니다. (아래 START 버튼을 누르세요)")
        st.caption("※ 카메라가 켜지지 않으면 `localhost` 또는 `https` 접속인지 확인하세요.")
        webrtc_streamer(
            key="bin-live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    elif mode == "🖼️ 이미지 업로드":
        uploaded = st.file_uploader(
            "폐기물 사진을 촬영하거나 이미지를 업로드하세요 (jpg / jpeg / png)",
            type=["jpg", "jpeg", "png"],
            help="휴대폰에서는 업로드 버튼을 누르면 '사진 촬영'을 선택할 수 있습니다.",
        )
        if uploaded:
            pil = Image.open(uploaded).convert("RGB")
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            run_single_image(bgr, frame_ph, result_ph, guide_ph)
            render_feedback_widget(backend)
        else:
            frame_ph.info("📸 사진을 촬영하거나 이미지를 업로드하면 즉시 분석합니다.\n\n"
                          "휴대폰에서는 업로드 버튼 → '사진 촬영'으로 바로 찍을 수 있습니다.")
            render_result_panel([], 0.0, result_ph, guide_ph, dev, show_guide)

    elif mode == "📷 사진 촬영":
        st.caption("※ 카메라가 뜨지 않으면 `localhost` 또는 `https` 접속인지 확인하세요. "
                   "(안드로이드는 업로드 모드보다 이 모드에서 카메라가 더 잘 뜹니다)")
        shot = st.camera_input("폐기물을 촬영하세요 📸")
        if shot is not None:
            pil = Image.open(shot).convert("RGB")
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            run_single_image(bgr, frame_ph, result_ph, guide_ph)
            render_feedback_widget(backend)
        else:
            frame_ph.info("📸 카메라로 폐기물을 촬영하면 즉시 분석합니다.")
            render_result_panel([], 0.0, result_ph, guide_ph, dev, show_guide)

    # ── 하단 — 개발자 모드: 통계 + 성능 모니터 ──────────────────────────────
    if dev:
        st.divider()
        dash_l, dash_r = st.columns([1, 1], gap="large")
        with dash_l:
            st.markdown("#### 📊 오늘의 탐지 통계")
            counts = st.session_state.counts
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("전체", f"{st.session_state.total}건")
            m2.metric("페트병", f"{counts.get('pet_bottle', 0)}건")
            m3.metric("비닐", f"{counts.get('vinyl', 0)}건")
            m4.metric("캔", f"{counts.get('can', 0)}건")
            m5.metric("종이", f"{counts.get('paper', 0)}건")
            if st.session_state.total > 0:
                import pandas as pd
                df = pd.DataFrame({"분류": [config.CLASS_NAMES_KO.get(k, k) for k in counts.keys()],
                                   "건수": list(counts.values())})
                st.bar_chart(df.set_index("분류"), color="#16a34a")
        with dash_r:
            st.markdown("#### ⚡ 추론 성능 모니터")
            lat = st.session_state.infer_ms
            if lat:
                import pandas as pd
                avg, mn, mx = sum(lat)/len(lat), min(lat), max(lat)
                p1, p2, p3 = st.columns(3)
                p1.metric("평균 지연", f"{avg:.0f} ms")
                p2.metric("최소", f"{mn:.0f} ms")
                p3.metric("최대", f"{mx:.0f} ms")
                st.line_chart(pd.DataFrame({"추론시간(ms)": lat[-50:]}), color="#3b82f6")
                st.caption(f"평균 처리량 ≈ {1000/avg:.0f} FPS · 최근 {len(lat[-50:])} 프레임")
            else:
                st.info("추론을 실행하면 지연 시간 그래프가 표시됩니다.")

# ── TAB 2: 분리배출 가이드 (QnA) ─────────────────────────────────────────────
with tab2:
    st.markdown("## 📚 자주 묻는 분리배출 가이드")

    st.markdown("### ♻️ 기본 원칙")
    st.info("내용물을 비운다 → 이물질 제거 → 재질별 분리 → 오염된 것은 일반쓰레기")

    st.markdown("### 🧴 스티로폼")
    st.warning("✔ 깨끗하면 재활용 가능 / ✖ 음식물 묻으면 일반쓰레기")

    st.markdown("### 🥤 종이컵")
    st.warning("✔ 종이처럼 보여도 코팅됨 / ✖ 대부분 일반쓰레기")

    st.markdown("### 📦 골판지 상자")
    st.warning("✔ 테이프 제거 후 종이류 배출 / ✖ 보냉 택배 상자(비닐·알루미늄 코팅)는 일반쓰레기")

    st.markdown("### 🧸 플라스틱 & 장난감")
    st.warning("✖ 옷걸이·칫솔·CD/DVD → 일반쓰레기 / ✖ 배터리 포함 완구 → 전자제품 수거함")

    st.markdown("### 🥡 배달용기")
    st.warning("✔ 재질보다 '세척 여부'가 핵심 / ✖ 음식물 묻으면 재활용 불가")

    st.markdown("### 🧾 유리병")
    st.success("✔ 내용물 제거 후 병류 배출 / ✔ 뚜껑 분리")

    st.markdown("---")
    st.markdown(
        "- **분리의 정석** — 품목별 배출방법: "
        "[바로가기](https://xn--oy2b29bd3a601b.kr/front/dischargeMethod/typeItem.do?searchCnd=1105)\n"
        "- **서울 도봉구 자원순환과**: "
        "[도봉구 청소행정](https://www.dobong.go.kr/subsite/waste/Contents.asp?code=10007358) "
        "☎ 02-2091-3262"
    )
    st.caption("※ 분리배출 규칙은 지자체·주거형태에 따라 다를 수 있으니 거주지 규정을 확인하세요.")