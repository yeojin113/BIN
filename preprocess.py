# preprocess.py
# ─────────────────────────────────────────────────────────────────────────────
# AI Hub 생활 폐기물 데이터셋 → YOLOv8 탐지 학습 형식 변환 (수정판)
#
# 이전 버전의 문제와 수정 내용:
#   1) 라벨 확장자가 ".Json"(대문자 J)인데 ".json"만 찾아 전부 누락
#      → 대소문자 무시 매칭으로 16만 라벨 모두 사용
#   2) 라벨이 bbox가 아니라 Bounding[].PolygonPoint(폴리곤)였음
#      → 폴리곤 min/max 로 bounding box 생성 (전체화면 박스 폐기)
#   3) 프레임 단위 random split → data leakage
#      → 촬영 ID(접두 포함, 예: 22_X005_C187_1105) 단위 split
#   4) 라벨 못 찾으면 전체화면(0.5,0.5,1,1) 폴백 → 위치 학습 무력화
#      → 폴백 제거. 라벨 없으면 스킵 + 카운트
#
# 사용법:
#   python preprocess.py
#   python preprocess.py --val_ratio 0.2
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2


# ══════════════════════════════════════════════════════════════════════════════
# 클래스 매핑 (폴더명에 키워드 포함 시 분류)
# ══════════════════════════════════════════════════════════════════════════════

CLASS_MAP = {
    "페트병": 0,    # pet_bottle
    "비닐":   1,    # vinyl
    "캔류":   2,    # can
    "종이류": 3,    # paper
}
CLASS_NAMES = ["pet_bottle", "vinyl", "can", "paper"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# 촬영 그룹 키: 접두까지 포함 (예: 22_X005_C187_1105)
#   - 같은 물체의 프레임(_0,_1,...)은 같은 키 → 한쪽 split 에만 배치(leakage 차단)
#   - 접두(15/21/22)가 다르면 다른 촬영 → 다른 키
GROUP_RE = re.compile(r"^(\d+_X\d+_C\d+_\d+)", re.IGNORECASE)


def get_class_id(folder_name: str):
    for keyword, cls_id in CLASS_MAP.items():
        if keyword in folder_name:
            return cls_id
    return None


def group_key(stem: str) -> str:
    """파일 stem에서 촬영 그룹 키 추출. 실패 시 stem 전체 사용."""
    m = GROUP_RE.match(stem)
    return m.group(1).upper() if m else stem


# ══════════════════════════════════════════════════════════════════════════════
# 폴리곤 JSON → YOLO bbox 변환
# ══════════════════════════════════════════════════════════════════════════════

def _resolution_from_json(data: dict):
    """RESOLUTION "1921*1440" → (1921, 1440). 실패 시 None."""
    res = data.get("RESOLUTION")
    if not res:
        return None
    try:
        w, h = re.split(r"[*xX]", res.strip())
        return int(w), int(h)
    except Exception:
        return None


def _points_minmax(polygon_point_list):
    """[{'Point1':'733,159'}, ...] → (xs, ys). 키 이름 무시, 값만 사용."""
    xs, ys = [], []
    for d in polygon_point_list:
        for v in d.values():
            try:
                xs_, ys_ = v.split(",")
                xs.append(float(xs_)); ys.append(float(ys_))
            except Exception:
                continue
    return xs, ys


def json_to_yolo(json_path: Path, class_id: int, img_w: int, img_h: int,
                 use_image_size: bool = True):
    """
    AI Hub JSON → YOLO 라인 리스트.
    두 가지 어노테이션 형식 모두 지원:
      - Drawing="BOX"     : x1,y1,x2,y2 (좌상단·우하단) 직접 사용
      - Drawing="POLYGON" : PolygonPoint 점들의 min/max 외접 사각형
    다중 객체(Bounding 여러 개)도 처리.

    use_image_size=True 이면 정규화 분모로 실제 이미지 크기(img_w,img_h)만 사용.
    (좌표는 실제 픽셀 기준이므로 이미지 크기가 정답. JSON RESOLUTION 불신.)
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"    [WARN] JSON 읽기 실패: {json_path.name} → {e}")
        return []

    if use_image_size:
        W, H = img_w, img_h
    else:
        res = _resolution_from_json(data)
        W, H = res if res else (img_w, img_h)
    if not W or not H:
        return []

    boundings = data.get("Bounding", [])
    if isinstance(boundings, dict):
        boundings = [boundings]

    lines = []
    for b in boundings:
        if not isinstance(b, dict):
            continue

        # ── 박스 좌표 추출: Drawing 형식에 따라 BOX / POLYGON 모두 지원 ──────
        drawing = (b.get("Drawing") or "").upper()
        x1 = y1 = x2 = y2 = None

        if "POLYGON" in drawing or b.get("PolygonPoint"):
            # 폴리곤: 점들의 min/max 로 외접 사각형
            pts = b.get("PolygonPoint")
            if pts:
                xs, ys = _points_minmax(pts)
                if len(xs) >= 2:
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)
        if x1 is None and (b.get("x1") is not None or "BOX" in drawing):
            # BOX: x1,y1,x2,y2 (좌상단·우하단) 를 그대로 사용
            try:
                x1 = float(b["x1"]); y1 = float(b["y1"])
                x2 = float(b["x2"]); y2 = float(b["y2"])
            except (KeyError, ValueError, TypeError):
                x1 = y1 = x2 = y2 = None

        if x1 is None:          # 어떤 형식도 못 읽음 → 스킵
            continue

        # min/max 정렬 보장 (x1>x2 같은 역순 대비)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # (지적 3) 정규화 '전'에 좌표를 이미지 경계로 클리핑.
        #          AI Hub 데이터에 음수/초과 좌표가 섞여 있어도 박스가 왜곡되지 않도록.
        x1 = max(0.0, min(float(W), x1))
        x2 = max(0.0, min(float(W), x2))
        y1 = max(0.0, min(float(H), y1))
        y2 = max(0.0, min(float(H), y2))
        if x2 <= x1 or y2 <= y1:        # 클리핑 후 무효 박스 제외
            continue

        cx = (x1 + x2) / 2 / W
        cy = (y1 + y2) / 2 / H
        nw = (x2 - x1) / W
        nh = (y2 - y1) / H

        if nw <= 0 or nh <= 0:
            continue
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return lines


# ══════════════════════════════════════════════════════════════════════════════
# 라벨 인덱스 (stem → .Json 경로) 사전 구축
# ══════════════════════════════════════════════════════════════════════════════

def build_label_index(label_root: Path) -> dict:
    """
    라벨 폴더 전체를 한 번 훑어 {stem(소문자): json_path} 인덱스 생성.
    매 이미지마다 rglob 하면 느리므로 1회 구축해 O(1) 조회.
    .Json 대소문자 무시.
    """
    index = {}
    if not label_root or not label_root.exists():
        return index
    for p in label_root.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".json":
            index[p.stem.lower()] = p
    return index


# ══════════════════════════════════════════════════════════════════════════════
# 메인 전처리
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(root_dir: Path, output_dir: Path, val_ratio: float = 0.2, seed: int = 42):
    random.seed(seed)

    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    train_root = root_dir / "Training"
    if not train_root.exists():
        print(f"[ERROR] Training 폴더 없음: {train_root}")
        return

    label_root = train_root / "Training_라벨링데이터"
    print(f"\n[1/4] 라벨 인덱스 구축 중... ({label_root})")
    label_index = build_label_index(label_root)
    print(f"      .Json 라벨 {len(label_index)}개 인덱싱 완료")

    # ── 원천(이미지) 폴더 순회: stem → (이미지경로, 클래스) 수집 ──────────────
    print(f"[2/4] 이미지 수집 + 클래스 매핑 중...")
    class_folders = [
        d for d in train_root.iterdir()
        if d.is_dir() and "라벨링" not in d.name
    ]

    # 그룹 단위로 모으기: group_key → [(img_path, cls_id, stem), ...]
    # (지적 2) split 왜곡 방지 — 라벨이 있는 이미지만 그룹에 포함시킨다.
    #          라벨 없는 이미지는 어차피 학습에 안 쓰이므로 split 계산에서도 제외.
    groups = defaultdict(list)
    n_img_total = 0
    n_img_labeled = 0
    no_label_by_class = defaultdict(int)   # 라벨 누락의 클래스별 분포
    for cf in sorted(class_folders):
        cls_id = get_class_id(cf.name)
        if cls_id is None:
            print(f"      [SKIP] 매핑 없음: {cf.name}")
            continue
        for f in cf.rglob("*"):
            if f.suffix.lower() in IMAGE_EXTS and f.is_file():
                n_img_total += 1
                if f.stem.lower() in label_index:
                    groups[group_key(f.stem)].append((f, cls_id, f.stem))
                    n_img_labeled += 1
                else:
                    no_label_by_class[cls_id] += 1
    print(f"      전체 이미지 {n_img_total}장 / 라벨 있는 이미지 {n_img_labeled}장 "
          f"/ 촬영 그룹 {len(groups)}개")

    # (지적 1) 다중 클래스 그룹 진단 — 단일 클래스 가정이 맞는지 데이터로 확인
    multiclass = 0
    for gk, members in groups.items():
        if len({m[1] for m in members}) > 1:
            multiclass += 1
    print(f"      다중 클래스 그룹: {multiclass}개  (0이면 '촬영=단일 클래스' 가정 확정)")

    # (지적 2) 라벨 누락의 클래스별 분포 — stratification 왜곡 가늠
    if any(no_label_by_class.values()):
        print("      라벨 누락(클래스별):", end=" ")
        for i, nm in enumerate(CLASS_NAMES):
            print(f"{nm}={no_label_by_class.get(i,0)}", end="  ")
        print()

    # ── 촬영 그룹 단위 train/val split ────────────────────────────────────────
    print(f"[3/4] 촬영 ID 단위 train/val 분할 (val={val_ratio:.0%})...")
    from collections import Counter
    by_class_groups = defaultdict(list)
    for gk, members in groups.items():
        # (지적 1) 첫 항목 대신 다수결 클래스를 대표로 (다중 클래스 그룹 안전장치)
        rep_cls = Counter(m[1] for m in members).most_common(1)[0][0]
        by_class_groups[rep_cls].append(gk)

    val_groups = set()
    for cls_id, gks in by_class_groups.items():
        random.shuffle(gks)
        n_val = int(round(len(gks) * val_ratio))
        if len(gks) > 1:
            n_val = max(1, n_val)        # 클래스별 val 최소 1그룹 보장
        val_groups.update(gks[:n_val])

    # (지적 3) 그룹 수 기준 split이 실제 '이미지 수' 기준으로 얼마나 쏠리는지 미리 출력
    img_count = {"train": defaultdict(int), "val": defaultdict(int)}
    for gk, members in groups.items():
        sp = "val" if gk in val_groups else "train"
        for _, cls_id, _ in members:
            img_count[sp][cls_id] += 1
    print(f"      [예상 분포 — 실제 이미지 기준]")
    print(f"      {'클래스':<12}{'train':>9}{'val':>9}{'val%':>7}")
    for i, nm in enumerate(CLASS_NAMES):
        tr, va = img_count['train'].get(i, 0), img_count['val'].get(i, 0)
        pct = va / max(tr + va, 1) * 100
        flag = "  ⚠️" if (tr + va > 0 and (pct < 5 or pct > 40)) else ""
        print(f"      {nm:<12}{tr:>9}{va:>9}{pct:>6.1f}%{flag}")

    # ── 이미지/라벨 생성 ──────────────────────────────────────────────────────
    print(f"[4/4] 이미지 복사 + 폴리곤→bbox 라벨 생성 중...")
    stats = {"train": 0, "val": 0, "no_label": 0, "no_box": 0,
             "bad_img": 0, "res_mismatch": 0}

    for gk, members in groups.items():
        split = "val" if gk in val_groups else "train"
        for img_path, cls_id, stem in members:
            _process_one(img_path, cls_id, stem, split, output_dir, label_index, stats)

    # ── 결과 ──────────────────────────────────────────────────────────────────
    print("\n" + "═" * 58)
    print("  전처리 완료")
    print("═" * 58)
    print(f"  train 이미지     : {stats['train']}장")
    print(f"  val   이미지     : {stats['val']}장")
    print(f"  라벨 없음(스킵)  : {stats['no_label']}장")
    print(f"  유효 박스 없음    : {stats['no_box']}장")
    print(f"  이미지 로드 실패  : {stats['bad_img']}장")
    print(f"  해상도 불일치     : {stats['res_mismatch']}장  "
          f"(JSON↔실제 이미지 크기 다름 — 이미지 크기로 보정함)")
    print("═" * 58)

    _write_yaml(output_dir)


def _process_one(img_path, cls_id, stem, split, output_dir, label_index, stats):
    # 라벨 먼저 확인 (없으면 이미지도 만들지 않음 → 전체화면 폴백 제거)
    json_path = label_index.get(stem.lower())
    if json_path is None:
        stats["no_label"] += 1
        return

    img = cv2.imread(str(img_path))
    if img is None:
        stats["bad_img"] += 1
        return
    img_h, img_w = img.shape[:2]

    # (지적 4) JSON 해상도와 실제 이미지 크기 불일치 점검.
    #          폴리곤 좌표는 실제 픽셀 기준이므로, 정규화 분모는 '실제 이미지 크기'가
    #          정답이다. JSON RESOLUTION 은 참고만 하고, 불일치 시 카운트.
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            jdata = json.load(f)
        jres = _resolution_from_json(jdata)
        if jres and (jres[0] != img_w or jres[1] != img_h):
            stats["res_mismatch"] += 1
    except Exception:
        pass

    # 정규화는 항상 실제 이미지 크기(img_w, img_h) 기준
    yolo_lines = json_to_yolo(json_path, cls_id, img_w, img_h, use_image_size=True)
    if not yolo_lines:
        stats["no_box"] += 1
        return

    out_img = output_dir / "images" / split / f"{stem}.jpg"
    out_lbl = output_dir / "labels" / split / f"{stem}.txt"

    if img_path.suffix.lower() in {".jpg", ".jpeg"}:
        shutil.copy(img_path, out_img)
    else:
        cv2.imwrite(str(out_img), img)

    with open(out_lbl, "w", encoding="utf-8") as f:
        f.write("\n".join(yolo_lines) + "\n")

    stats[split] += 1


def _write_yaml(output_dir: Path):
    yaml_path = output_dir / "dataset.yaml"
    content = (
        f"# dataset.yaml (preprocess.py 자동 생성 — 폴리곤→bbox, 촬영ID split)\n"
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n"
    )
    for i, nm in enumerate(CLASS_NAMES):
        content += f"  {i}: {nm}\n"
    yaml_path.write_text(content, encoding="utf-8")
    print(f"\n  dataset.yaml 생성: {yaml_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 진입점
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="AI Hub → YOLOv8 전처리 (폴리곤→bbox)")
    p.add_argument("--src", type=Path, default=Path("data"))
    p.add_argument("--dst", type=Path, default=Path("data"))
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.src.exists():
        print(f"[ERROR] 폴더 없음: {args.src}")
        exit(1)
    preprocess(args.src, args.dst, args.val_ratio, args.seed)