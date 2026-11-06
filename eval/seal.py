"""
[양세윤 작성 · test 담당 실행] test 봉인 — 저장소 밖 test 자료의 해시를 report/test_manifest.csv에 남긴다

봉인 대상 4종(모두 저장소 밖, test 담당 보관 — docs/plan.md 3.3 · 9절 단계 4):
  label   정답 폴더의 *.json (100% 2차 검수를 마친 것)
  text    1~3항 텍스트 폴더의 *.txt + _cut_log.csv(있으면)
  subset  서식 목록 CSV (doc_id, subset)      — eval/score.py --subset-csv
  group   제조사 그룹 CSV (doc_id, group)     — eval/paired.py --groups-csv (관계사는 같은 그룹)

manifest에는 파일 이름을 적지 않는다(doc_id에 제조사 약칭이 들어 있음, CLAUDE.md 공개 저장소 규칙).
  kind,sha256
  label,<파일 내용 해시>        파일마다 1줄, 해시 순 정렬
  label_set,<이름·해시 목록의 해시>  종류마다 1줄. 파일 이름이 바뀌거나 섞여도 잡는다
화면에도 건수 · 서식별 건수만 낸다. 어느 파일이 어긋났는지는 --show-names로만 본다(test 담당 전용).

  # test 담당: 봉인(manifest가 비어 있을 때 1회) → PM 승인 → manifest 커밋 → tag test-sealed
  python3 eval/seal.py --write  --label-dir <정답> --text-dir <텍스트> --subset-csv <서식> --groups-csv <그룹>
  python3 eval/seal.py --verify --label-dir <정답> --text-dir <텍스트> --subset-csv <서식> --groups-csv <그룹>

--manifest: 다른 test 세트를 따로 봉인한다(기본 report/test_manifest.csv = 팀 test, 동작 불변).
  새 test(cv/prereg_newtest.md)는 --manifest cv/newtest/test_manifest.csv. infer · score · paired도 같은 옵션을 받는다.

eval/infer.py(text) · eval/score.py(label · text · subset) · eval/paired.py(group)는 test를 돌리기 전에
require_sealed()로 넘겨받은 자료가 봉인과 같은지 확인하고, 다르거나 봉인 전이면 거부한다.
"""
import argparse
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pydantic import ValidationError  # noqa: E402

from src.schema import MSDSLabel  # noqa: E402

MANIFEST = ROOT / "report" / "test_manifest.csv"
KINDS = ("label", "text", "subset", "group")
CUT_LOG = "_cut_log.csv"
CSV_COLUMNS = {"subset": "subset", "group": "group"}  # kind → (doc_id 말고) 필요한 열


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def outside_repo(path):
    try:
        Path(path).resolve().relative_to(ROOT)
        return False
    except ValueError:
        return True


def file_hashes(kind, path):
    """→ {이름: 내용 해시}. 폴더(label · text)는 파일마다, CSV(subset · group)는 파일 1개(이름은 비워 둠 —
    넘길 때 파일 이름이 바뀌어도 내용이 같으면 같은 봉인)."""
    p = Path(path)
    if kind in CSV_COLUMNS:
        if not p.is_file():
            sys.exit(f"[거부] {kind} 파일이 없다: {p}")
        return {"": sha256_bytes(p.read_bytes())}
    if not p.is_dir():
        sys.exit(f"[거부] {kind} 폴더가 없다: {p}")
    files = sorted(p.glob("*.json" if kind == "label" else "*.txt"))
    if kind == "text" and (p / CUT_LOG).exists():
        files.append(p / CUT_LOG)
    return {f.name: sha256_bytes(f.read_bytes()) for f in files}


def set_hash(hashes):
    return sha256_bytes("\n".join(f"{name}\t{h}" for name, h in sorted(hashes.items())).encode("utf-8"))


def manifest_rows(kind, hashes):
    return [(kind, h) for h in sorted(hashes.values())] + [(f"{kind}_set", set_hash(hashes))]


def read_manifest(path=None):
    path = Path(path or MANIFEST)
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and set(rows[0]) != {"kind", "sha256"}:
        sys.exit(f"[거부] {path}의 열이 kind,sha256이 아니다")
    return [(r["kind"], r["sha256"]) for r in rows]


def rel_manifest(manifest_path=None):
    p = Path(manifest_path or MANIFEST).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def verify(kind, path, manifest=None, manifest_path=None):
    """→ (맞는지, 사유 목록, 어긋난 파일 이름). 이름은 호출한 쪽이 --show-names일 때만 보여 준다."""
    rows = read_manifest(manifest_path) if manifest is None else manifest
    sealed = Counter(h for k, h in rows if k == kind)
    sealed_set = [h for k, h in rows if k == f"{kind}_set"]
    if not sealed or len(sealed_set) != 1:
        return False, [f"{kind}: 봉인 기록 없음(eval/seal.py --write 전)"], []
    hashes = file_hashes(kind, path)
    now = Counter(hashes.values())
    problems, names = [], []
    extra, lacking = now - sealed, sealed - now
    if extra or lacking:
        problems.append(f"{kind}: 봉인과 다른 파일 {sum(extra.values())}개, 봉인에 있는데 없는 파일 {sum(lacking.values())}개")
        names = sorted(n or Path(path).name for n, h in hashes.items() if h in extra)
    elif set_hash(hashes) != sealed_set[0]:
        problems.append(f"{kind}: 내용은 같지만 파일 이름이 봉인과 다르다")
    return not problems, problems, names


def require_sealed(kind, path, manifest_path=None):
    """test 실행 전 게이트. 봉인 전이거나 봉인과 다르면 멈춘다(이름은 내지 않음)."""
    ok, problems, _ = verify(kind, path, manifest_path=manifest_path)
    if not ok:
        sys.exit(f"[거부] test {kind} 자료가 봉인({rel_manifest(manifest_path)})과 맞지 않다: {'; '.join(problems)}. "
                 "test 담당이 eval/seal.py --verify --show-names로 확인할 것")


# ---------------------------------------------------------------- 봉인 전 점검(--write)
def read_id_csv(kind, path):
    """(doc_id, subset|group) CSV → {doc_id: 값}. 중복 · 빈 값이면 거부."""
    col = CSV_COLUMNS[kind]
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"doc_id", col} <= set(rows[0]):
        sys.exit(f"[거부] {kind} CSV에는 doc_id,{col} 열이 있어야 한다")
    out = {}
    for r in rows:
        if r["doc_id"] in out or not r["doc_id"] or not r[col].strip():
            sys.exit(f"[거부] {kind} CSV에 doc_id 중복 또는 빈 값이 있다")
        out[r["doc_id"]] = r[col].strip()
    return out


def cut_not_found(text_dir):
    p = Path(text_dir) / CUT_LOG
    if not p.exists():
        return set()
    with open(p, encoding="utf-8-sig", newline="") as f:
        return {r["doc_id"] for r in csv.DictReader(f) if r.get("doc_id") and r.get("status") == "NOT_FOUND"}


def check_before_write(paths, show_names):
    """정답 스키마 · 네 자료의 문서 목록 일치. 문제가 있으면 건수만 내고 멈춘다(--show-names면 이름도)."""
    labels = sorted(Path(paths["label"]).glob("*.json"))
    bad = []
    for f in labels:
        try:
            MSDSLabel.model_validate_json(f.read_text(encoding="utf-8"))
        except ValidationError:
            bad.append(f.stem)
    ids = {f.stem for f in labels}
    subset = read_id_csv("subset", paths["subset"])
    group = read_id_csv("group", paths["group"])
    texts = {f.stem for f in Path(paths["text"]).glob("*.txt")} | cut_not_found(paths["text"])
    problems = []
    if not ids:
        problems.append(("정답 파일 0개", []))
    if bad:
        problems.append((f"스키마를 어긴 정답 {len(bad)}건(scripts/validate_labels.py로 확인)", bad))
    for name, got in (("서식 목록", set(subset)), ("그룹 목록", set(group)), ("텍스트(+_cut_log NOT_FOUND)", texts)):
        diff = sorted(ids ^ got)
        if diff:
            problems.append((f"정답과 {name}의 문서가 {len(diff)}건 다르다", diff))
    if problems:
        for msg, names in problems:
            print(f"  - {msg}" + (f": {names}" if show_names and names else ""))
        sys.exit("[거부] 봉인 전 점검 실패 — 고친 뒤 다시 --write")
    return len(ids), Counter(subset.values()), len(set(group.values()))


def write(paths, show_names, manifest_path=None):
    manifest_path = Path(manifest_path or MANIFEST)
    if any(k in KINDS for k, _ in read_manifest(manifest_path)):
        sys.exit(f"[거부] {manifest_path}에 이미 봉인이 있다. 봉인은 1회만 한다(바꾸려면 PM 결정 + report/decisions.md 기록)")
    n, forms, n_groups = check_before_write(paths, show_names)
    rows = [r for kind in KINDS for r in manifest_rows(kind, file_hashes(kind, paths[kind]))]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["kind", "sha256"])
        w.writerows(rows)
    print(f"봉인: 정답 {n}건 · 제조사 그룹 {n_groups}개 · 서식별 " + " · ".join(f"{k} {v}" for k, v in sorted(forms.items())))
    print(f"→ {manifest_path} (파일 이름 없음). PM 승인 뒤 커밋하고 tag test-sealed")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="봉인 해시를 manifest에 쓴다(봉인 전 1회)")
    mode.add_argument("--verify", action="store_true", help="지금 자료가 봉인과 같은지 대조")
    ap.add_argument("--label-dir", required=True)
    ap.add_argument("--text-dir", required=True)
    ap.add_argument("--subset-csv", required=True)
    ap.add_argument("--groups-csv", required=True)
    ap.add_argument("--show-names", action="store_true", help="어긋난 파일 이름을 화면에 낸다(test 담당 전용)")
    ap.add_argument("--manifest", help="봉인 기록 파일(기본 report/test_manifest.csv = 팀 test)")
    args = ap.parse_args(argv)

    paths = {"label": args.label_dir, "text": args.text_dir, "subset": args.subset_csv, "group": args.groups_csv}
    inside = [k for k, p in paths.items() if not outside_repo(p)]
    if inside:
        sys.exit(f"[거부] test 자료는 저장소 밖에 둔다: {inside}")
    if args.write:
        write(paths, args.show_names, args.manifest)
        return
    all_ok = True
    for kind in KINDS:
        ok, problems, names = verify(kind, paths[kind], manifest_path=args.manifest)
        all_ok &= ok
        print(f"{kind}: 일치" if ok else "\n".join(f"{kind}: 불일치 — {p}" for p in problems)
              + (f"\n    {names}" if args.show_names and names else ""))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
