# resplit.py
# ─────────────────────────────────────────────────────────────────────────────
# 촬영 ID 단위 train/val 재분할 (data leakage 제거) — Stratified Group Split
#
# 문제:
#   기존 split은 같은 촬영물(예: X001_C011)의 여러 프레임(_0,_1,_2...)이
#   train과 val에 흩어져 있어 data leakage 발생 → mAP 과대평가.
#
# 해결:
#   파일명에서 그룹 키 "{X}_{C}"(예: X001_C011)를 추출하고,
#   같은 그룹은 통째로 한쪽에만 배치(leakage 차단)하되,
#   클래스 비율은 최대한 보존(stratified)한다.
#
# 반영된 개선:
#   1) 그룹 대표 클래스 = 그룹 내 다수(majority) 클래스 (첫 항목 X)
#   2) StratifiedGroupKFold 로 진짜 stratified group split
#   3) 소수 클래스도 val에 최소 1그룹 보장 (그룹 1개뿐이면 예외 처리)
#   4) symlink 실패 시 copy2 fallback
#
# 안전:
#   - 기존 train/val 폴더는 건드리지 않음
#   - 새 폴더(images/train_clean, val_clean)에 심볼릭 링크 → 디스크 0 추가
#   - 그룹 겹침 0 검증 + 다중클래스 그룹 진단 + 클래스별 통계 출력
#
# 사용법:
#   python resplit.py                 # 미리보기(통계만)
#   python resplit.py --apply         # 실제 링크 + yaml 생성
#   python resplit.py --apply --val-ratio 0.1
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import config

# ── 경로 ──────────────────────────────────────────────────────────────────────
DATA = config.ROOT / "data"
SRC = [
    (DATA / "images" / "train", DATA / "labels" / "train"),
    (DATA / "images" / "val",   DATA / "labels" / "val"),
]
DST = {
    "train": (DATA / "images" / "train_clean", DATA / "labels" / "train_clean"),
    "val":   (DATA / "images" / "val_clean",   DATA / "labels" / "val_clean"),
}
OUT_YAML = DATA / "dataset_clean.yaml"

GROUP_RE = re.compile(r"(X\d+_C\d+)", re.IGNORECASE)


def group_key(filename: str) -> str | None:
    m = GROUP_RE.search(filename)
    return m.group(1).upper() if m else None


def read_primary_class(label_path: Path) -> int | None:
    """라벨 파일의 첫 박스 class_id (이미지당 1객체 가정)."""
    if not label_path.exists():
        return None
    try:
        with open(label_path) as f:
            for line in f:
                parts = line.split()
                if parts:
                    return int(parts[0])
    except Exception:
        return None
    return None


def collect():
    """모든 (이미지, 라벨, 그룹, 클래스) 수집."""
    items, missing = [], 0
    for img_dir, lbl_dir in SRC:
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            lbl = lbl_dir / (img.stem + ".txt")
            g = group_key(img.name)
            if g is None:
                missing += 1
                g = img.stem
            items.append((img, lbl, g, read_primary_class(lbl)))
    return items, missing


def build_groups(items):
    """그룹별 멤버 + 다수결 대표 클래스 + 다중클래스 진단."""
    group_members = defaultdict(list)          # g -> [(img, lbl), ...]
    group_class_counter = defaultdict(Counter)  # g -> Counter({cls: n})
    for img, lbl, g, cls in items:
        group_members[g].append((img, lbl))
        if cls is not None:
            group_class_counter[g][cls] += 1

    # (1) 다수 클래스를 대표로
    group_class = {}
    multiclass_groups = 0
    for g, ctr in group_class_counter.items():
        if not ctr:
            group_class[g] = -1
            continue
        if len(ctr) > 1:
            multiclass_groups += 1
        group_class[g] = ctr.most_common(1)[0][0]
    # 라벨이 하나도 없던 그룹 보정
    for g in group_members:
        group_class.setdefault(g, -1)

    return group_members, group_class, multiclass_groups


def stratified_group_split(group_members, group_class, val_ratio, seed):
    """
    임의 val_ratio 를 정확히 맞추는 hold-out group split + 클래스 비율 보존.

    설계 노트:
      - sklearn 에는 'stratified + group + shuffle' 을 한 번에 하는 클래스가 없다
        (StratifiedGroupShuffleSplit 은 존재하지 않음).
      - 따라서 그룹 분리와 정확한 비율은 GroupShuffleSplit(test_size=val_ratio)로
        보장하고, 클래스 균형은 '클래스별 GroupShuffleSplit' 방식으로 stratify 한다.
        즉 클래스마다 그 클래스 그룹들을 val_ratio 로 잘라 합치면,
        그룹 분리를 유지하면서 클래스 비율도 보존된다 (1그룹=1클래스 대표 기준).
      - sklearn 이 없으면 동일 논리를 random 셔플로 폴백.

    (3) 그룹이 2개 이상인 클래스는 val 최소 1그룹 보장.
    """
    by_class = defaultdict(list)
    for g in group_members:
        by_class[group_class[g]].append(g)

    assign = {}
    used_sklearn = False
    try:
        from sklearn.model_selection import GroupShuffleSplit
        used_sklearn = True
        for cls, gs in by_class.items():
            if len(gs) == 1:
                assign[gs[0]] = "train"          # 그룹 1개뿐 → train 고정
                continue
            gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
            # 각 그룹을 1개의 sample 로 보고, groups=자기자신 → 그룹 단위 분리
            idx = list(range(len(gs)))
            tr_idx, va_idx = next(gss.split(idx, groups=gs))
            va = set(va_idx)
            if len(va) == 0:                     # (3) 비율이 작아 0이면 최소 1 보장
                va = {va_idx[0]} if len(va_idx) else {idx[-1]}
            for i in idx:
                assign[gs[i]] = "val" if i in va else "train"
    except Exception as e:
        print(f"[폴백] GroupShuffleSplit 사용 불가 ({e.__class__.__name__}) → 클래스별 셔플 분배")
        random.seed(seed)
        for cls, gs in by_class.items():
            gs = gs[:]
            random.shuffle(gs)
            if len(gs) == 1:
                assign[gs[0]] = "train"
                continue
            n_val = max(1, int(round(len(gs) * val_ratio)))   # (3) 최소 1
            for g in gs[:n_val]:
                assign[g] = "val"
            for g in gs[n_val:]:
                assign[g] = "train"

    return assign, used_sklearn


def summarize(group_members, group_class, assign, multiclass_groups, missing):
    counts = {"train": defaultdict(int), "val": defaultdict(int)}
    groups_in = {"train": set(), "val": set()}
    for g, members in group_members.items():
        s = assign[g]
        groups_in[s].add(g)
        counts[s][group_class[g]] += len(members)

    overlap = groups_in["train"] & groups_in["val"]
    names = config.CLASS_NAMES
    tr_total = sum(counts["train"].values())
    va_total = sum(counts["val"].values())

    print("\n" + "═" * 60)
    print("  재분할 결과 (촬영 ID 단위 · Stratified Group Split)")
    print("═" * 60)
    print(f"  {'클래스':<14}{'train':>10}{'val':>10}{'val%':>8}")
    print("  " + "-" * 42)
    for i, nm in enumerate(names):
        tr, va = counts['train'].get(i, 0), counts['val'].get(i, 0)
        pct = (va / max(tr + va, 1)) * 100
        flag = "  ⚠️ val 0" if va == 0 else ""
        print(f"  {nm:<14}{tr:>10}{va:>10}{pct:>7.1f}%{flag}")
    unk_tr, unk_va = counts['train'].get(-1, 0), counts['val'].get(-1, 0)
    if unk_tr or unk_va:
        print(f"  {'(라벨없음)':<14}{unk_tr:>10}{unk_va:>10}")
    print("  " + "-" * 42)
    print(f"  {'합계':<14}{tr_total:>10}{va_total:>10}{va_total/max(tr_total+va_total,1)*100:>7.1f}%")

    print(f"\n  그룹 수            : train {len(groups_in['train'])} | val {len(groups_in['val'])}")
    print(f"  다중 클래스 그룹   : {multiclass_groups}개  (0이면 '촬영당 단일 클래스' 구조 확정)")
    print(f"  그룹 추출 실패     : {missing}장")
    print(f"  ⚠️ train/val 그룹 겹침: {len(overlap)}  (0이어야 정상)")
    print("═" * 60)
    return len(overlap) == 0


def _link_or_copy(src: Path, dst: Path):
    """(4) symlink 우선, 실패 시 copy2 폴백."""
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def apply_links(group_members, assign):
    for split, (img_d, lbl_d) in DST.items():
        for d in (img_d, lbl_d):
            if d.exists():
                for f in d.iterdir():
                    f.unlink()
            d.mkdir(parents=True, exist_ok=True)

    n = 0
    for g, members in group_members.items():
        img_d, lbl_d = DST[assign[g]]
        for img, lbl in members:
            _link_or_copy(img, img_d / img.name)
            if lbl.exists():
                _link_or_copy(lbl, lbl_d / lbl.name)
            n += 1
    print(f"\n[배치] {n}개 이미지(+라벨) 링크/복사 완료")


def write_yaml():
    content = (
        f"# dataset_clean.yaml (resplit.py 자동 생성 — Stratified Group Split)\n"
        f"path: {DATA}\n"
        f"train: images/train_clean\n"
        f"val:   images/val_clean\n"
        f"nc: {len(config.CLASS_NAMES)}\n"
        f"names:\n"
    )
    for i, nm in enumerate(config.CLASS_NAMES):
        content += f"  {i}: {nm}\n"
    OUT_YAML.write_text(content, encoding="utf-8")
    print(f"[YAML] 생성: {OUT_YAML}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("[수집] 이미지/라벨 스캔 중...")
    items, missing = collect()
    print(f"  총 {len(items)}장 수집")

    group_members, group_class, multiclass = build_groups(items)
    assign, used_sklearn = stratified_group_split(
        group_members, group_class, args.val_ratio, args.seed
    )
    print(f"  분할 방식: {'클래스별 GroupShuffleSplit (sklearn)' if used_sklearn else '클래스별 셔플 폴백'}")

    ok = summarize(group_members, group_class, assign, multiclass, missing)
    if not ok:
        print("\n[중단] 그룹 겹침 발견 → 링크 생성 안 함.")
        return

    if args.apply:
        apply_links(group_members, assign)
        write_yaml()
        print("\n✅ 완료. config.py의 DATA_YAML을 dataset_clean.yaml로 바꾼 뒤 학습하세요:")
        print("   sed -i 's|dataset.yaml|dataset_clean.yaml|' ~/BIN/config.py")
        print("   python train.py --no-augment --epochs 100 --batch 16 --device 0")
    else:
        print("\n[미리보기] 적용하려면 --apply 를 붙이세요.")


if __name__ == "__main__":
    main()