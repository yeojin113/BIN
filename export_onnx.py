# export_onnx.py
# ─────────────────────────────────────────────────────────────────────────────
# 학습된 YOLOv8 .pt 파일을 ONNX 포맷으로 변환합니다.
#
# 사용법:
#   python export_onnx.py
#   python export_onnx.py --weights models/waste_detector.pt --imgsz 640
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics 패키지가 없습니다: pip install ultralytics")
    sys.exit(1)

import config


def export_onnx(weights: Path, imgsz: int, opset: int, dynamic: bool, simplify: bool):
    """
    PyTorch 가중치(.pt)를 ONNX 형식으로 변환합니다.

    Args:
        weights:  .pt 파일 경로
        imgsz:    모델 입력 크기
        opset:    ONNX opset 버전 (기본 17)
        dynamic:  동적 배치 크기 지원 여부
        simplify: onnx-simplifier 적용 여부
    """
    if not weights.exists():
        print(f"[ERROR] 가중치 파일이 없습니다: {weights}")
        print("  먼저 train.py를 실행하여 모델을 학습하세요.")
        sys.exit(1)

    print(f"[Export] 모델 로드: {weights}")
    model = YOLO(str(weights))

    print(f"[Export] ONNX 변환 시작...")
    print(f"  imgsz:    {imgsz}")
    print(f"  opset:    {opset}")
    print(f"  dynamic:  {dynamic}")
    print(f"  simplify: {simplify}")

    exported_path = model.export(
        format   = "onnx",
        imgsz    = imgsz,
        opset    = opset,
        dynamic  = dynamic,
        simplify = simplify,
    )

    # 변환된 파일을 models/ 디렉토리로 이동
    exported = Path(exported_path)
    target   = config.MODEL_ONNX
    target.parent.mkdir(parents=True, exist_ok=True)

    if exported != target:
        import shutil
        shutil.copy(exported, target)
        print(f"[Export] 복사 완료: {exported} → {target}")

    print(f"\n[Export] 변환 성공!")
    print(f"  저장 위치: {target}")
    print(f"  파일 크기: {target.stat().st_size / 1024 / 1024:.1f} MB")

    # ── ONNX 모델 검증 ────────────────────────────────────────────────────────
    _verify_onnx(target)


def _verify_onnx(onnx_path: Path):
    """변환된 ONNX 모델의 유효성을 검증합니다."""
    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        print("[Verify] ONNX 모델 유효성 검증 통과 ✓")

        # 입출력 shape 출력
        print("\n  [입력]")
        for inp in model.graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"    {inp.name}: {shape}")

        print("  [출력]")
        for out in model.graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"    {out.name}: {shape}")

    except ImportError:
        print("[Verify] onnx 패키지 없음 → 검증 건너뜀 (pip install onnx)")
    except Exception as e:
        print(f"[Verify] 검증 실패: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 → ONNX 변환")
    parser.add_argument("--weights",  type=Path, default=config.MODEL_PT,  help=".pt 가중치 경로")
    parser.add_argument("--imgsz",   type=int,  default=config.IMG_SIZE,   help="입력 이미지 크기")
    parser.add_argument("--opset",   type=int,  default=17,                help="ONNX opset 버전")
    parser.add_argument("--dynamic", action="store_true", default=False,   help="동적 배치 크기 활성화")
    parser.add_argument("--simplify",action="store_true", default=True,    help="onnx-simplifier 적용")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_onnx(
        weights  = args.weights,
        imgsz    = args.imgsz,
        opset    = args.opset,
        dynamic  = args.dynamic,
        simplify = args.simplify,
    )