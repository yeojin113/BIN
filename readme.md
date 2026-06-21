# B:in (Bin + in)
### 실시간 웹 기반 폐기물 분류 시스템 · YOLOv8 + ONNX + Streamlit

---

## 1. 시스템 아키텍처

```
[카메라 / 이미지]
        ↓
  Letterbox 전처리
        ↓
  YOLOv8 탐지 모델 (ONNX Runtime)
        ↓
  NMS 후처리 (Conf + IoU 임계값)
        ↓
  BBox Smoothing (Moving Average)
        ↓
  Streamlit 웹 UI
        ↓
[클래스 표시 + 분리배출 안내 + 통계 대시보드]
```

---

## 2. 프로젝트 구조

```
BIN/
├── app.py               ← Streamlit 웹 앱 (메인 실행 파일)
├── train.py             ← YOLOv8 학습 스크립트
├── preprocess.py        ← AI Hub 데이터 → YOLOv8 형식 변환
├── export_onnx.py       ← .pt → .onnx 변환
├── inference_onnx.py    ← ONNX Runtime 추론
├── config.py            ← 전체 설정 중앙 관리
├── utils.py             ← 전처리 / 시각화 / 가이드
├── requirements.txt
├── readme.md
├── models/
│   ├── waste_detector.pt    ← 학습된 PyTorch 가중치
│   └── waste_detector.onnx  ← ONNX 변환 결과
├── data/
│   ├── dataset.yaml              ← YOLOv8 데이터셋 설정 (자동 업데이트)
│   ├── 생활 폐기물 이미지/        ← AI Hub 원본 데이터
│   │   ├── Training/
│   │   │   ├── [T원전]비닐_과자봉지_과자봉지/
│   │   │   ├── [T원전]종이류_노트_노트/
│   │   │   ├── [T원전]캔류_맥주캔_맥주캔/
│   │   │   ├── ...
│   │   │   └── Training_라벨링데이터/
│   │   └── Validation/
│   ├── images/          ← preprocess.py 실행 후 자동 생성
│   │   ├── train/
│   │   └── val/
│   └── labels/          ← preprocess.py 실행 후 자동 생성
│       ├── train/
│       └── val/
└── outputs/             ← 추론 결과 이미지 저장
```

---

## 3. 환경 설정 및 설치

### 3.1 Python 가상환경 생성 (권장)

```bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac / Linux
```

### 3.2 패키지 설치

```bash
pip install -r requirements.txt
```

> GPU 사용 시 `onnxruntime-gpu` 로 교체:
> ```bash
> pip uninstall onnxruntime
> pip install onnxruntime-gpu
> ```

---

## 4. 데이터 준비 (AI Hub)

### 4.1 데이터 다운로드

1. [AI Hub](https://aihub.or.kr) 접속 → 로그인
2. **생활 폐기물 이미지** 데이터셋 검색 후 다운로드
3. 압축 해제 후 `data/` 폴더 아래에 배치

```
data/
└── 생활 폐기물 이미지/
    ├── Training/
    └── Validation/
```

### 4.2 클래스 매핑

`preprocess.py`는 아래 규칙으로 폴더명을 클래스에 자동 매핑합니다.

| 폴더명 키워드 | 클래스 | ID |
|--------------|--------|----|
| 비닐 | vinyl | 1 |
| 종이류 | paper | 3 |
| 캔류 | can | 2 |
| (추후 추가) 페트병 | pet_bottle | 0 |

### 4.3 전처리 실행

```bash
python preprocess.py
```

실행 후 자동으로 생성되는 구조:

```
data/
├── images/
│   ├── train/   ← 이미지 복사 완료
│   └── val/     ← 이미지 복사 완료
├── labels/
│   ├── train/   ← YOLO .txt 라벨 변환 완료
│   └── val/     ← YOLO .txt 라벨 변환 완료
└── dataset.yaml ← 경로 자동 업데이트 완료
```

옵션 지정이 필요한 경우:

```bash
# 검증 비율 변경 (기본 20%)
python preprocess.py --val_ratio 0.15

# 원본 데이터 경로가 다를 경우
python preprocess.py --src "data/생활 폐기물 이미지" --dst data
```

---

## 5. 전체 실행 순서

```
1. pip install -r requirements.txt   ← 패키지 설치
2. python preprocess.py              ← 데이터 전처리
3. python train.py                   ← 모델 학습
4. python export_onnx.py             ← ONNX 변환
5. streamlit run app.py              ← 웹 앱 실행
```

---

## 6. 학습 방법

```bash
# 기본 학습 (Albumentations 증강 포함)
python train.py

# 파라미터 지정
python train.py --epochs 100 --batch 16 --imgsz 640 --device 0

# CPU 학습
python train.py --epochs 50 --batch 8 --device cpu

# 증강 없이 학습
python train.py --no-augment
```

학습 완료 후 `models/waste_detector.pt` 에 자동 저장됩니다.

학습 결과 (Precision / Recall / mAP) 는 터미널에 자동 출력됩니다.

---

## 7. ONNX 변환

```bash
python export_onnx.py

# 옵션 지정
python export_onnx.py --weights models/waste_detector.pt --imgsz 640 --opset 17
```

변환 완료 후 `models/waste_detector.onnx` 저장.

---

## 8. ONNX 추론 테스트

```bash
# 단일 이미지
python inference_onnx.py --source data/test.jpg

# 폴더 전체 배치 추론
python inference_onnx.py --source data/test_images/ --save

# 화면 표시 없이 저장만
python inference_onnx.py --source data/test.jpg --no-show --save
```

결과 이미지는 `outputs/` 폴더에 저장됩니다.

---

## 9. Streamlit 앱 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속.

### 앱 기능

| 기능 | 설명 |
|------|------|
| 이미지 업로드 | jpg / jpeg / png 업로드 후 즉시 분석 |
| 웹캠 실시간 | 카메라 연결 후 실시간 탐지 |
| Bounding Box | 클래스별 색상 구분 박스 |
| FPS 표시 | 실시간 추론 속도 |
| 분리배출 가이드 | 클래스별 올바른 배출 방법 안내 |
| 세션 통계 | 오늘의 클래스별 탐지 건수 집계 |
| NMS 조절 | Confidence / IoU 임계값 슬라이더 |

---

## 10. 클래스 & 분리배출 가이드

| 클래스 | 한국어 | 배출 방법 |
|--------|--------|-----------|
| pet_bottle | 페트병 | 라벨 제거 후 압착, 투명 페트병 전용함 |
| vinyl | 비닐 | 이물질 제거 후 비닐류 전용함 |
| can | 캔 | 내용물 비우고 캔 수거함 |
| paper | 종이 | 물기 제거 후 종이류 수거함 |

---

## 11. 성능 개선 아이디어

| 아이디어 | 설명 |
|----------|------|
| 모델 고도화 | YOLOv8n → YOLOv8s / m 으로 업그레이드 |
| 2차 분류 | 페트병 → 무색/유색 플라스틱 CNN 분류기 추가 |
| 데이터 증대 | 클래스별 500장 이상 확보, GAN 기반 합성 이미지 활용 |
| 엣지 배포 | TensorRT 또는 OpenVINO 변환으로 임베디드 디바이스 적용 |
| 다국어 지원 | 영어/중국어 분리배출 가이드 추가 (외국인 사용자) |
| 오염도 분석 | 컵·캔의 이물질 오염 여부 판별 기능 추가 |
| 통계 DB 연동 | SQLite 또는 Firebase로 통계 영구 저장 |

---

## 12. 자주 발생하는 오류

| 오류 | 해결 방법 |
|------|-----------|
| `FileNotFoundError: 생활 폐기물 이미지` | AI Hub 데이터를 `data/` 아래에 배치했는지 확인 |
| `FileNotFoundError: waste_detector.onnx` | `export_onnx.py` 먼저 실행 |
| `cv2.error: camera not found` | 웹캠 연결 확인, 장치 번호 0→1 변경 |
| `CUDA out of memory` | `--batch` 줄이거나 `--device cpu` 사용 |
| `albumentations import error` | `pip install albumentations` 실행 |
| `dataset.yaml 경로 오류` | `preprocess.py` 실행 후 자동 업데이트 확인 |