"""
[양세윤] 문서 목록 봉인 — 정답 · 텍스트를 만들기 전에 "어떤 문서가 최종 확인용인가"를 먼저 고정한다.

eval/seal.py는 정답 · 텍스트 · 서식 · 그룹 4종을 봉인한다. 그 4종은 문서를 열어야 만들 수 있으므로,
그 전에 원본 PDF 목록만 비밀 키 해시(HMAC-SHA256)로 봉인해 둔다. 커밋 시각이 문서 열람 · 라벨링보다 앞선다는 것이
확인용 문서를 결과에 맞춰 고르지 않았다는 근거다.

  python3 eval/seal_docs.py --write  --pdf-dir <저장소 밖 PDF 폴더> --key-file <저장소 밖 키> --manifest cv/holdout/docs_manifest.csv
  python3 eval/seal_docs.py --verify --pdf-dir <저장소 밖 PDF 폴더> --key-file <저장소 밖 키> --manifest cv/holdout/docs_manifest.csv

- manifest: kind,sha256 · 줄 이름 pdf_hmac(파일마다, 해시 순) · pdf_hmac_set(이름 · 해시 목록의 해시). 파일 이름은 적지 않는다.
- 키 없이 만든 평문 해시는 쓰지 않는다. 같은 PDF를 가진 사람이 어떤 문서가 확인용인지 맞춰 볼 수 있기 때문이다.
- 봉인은 1회만 한다. 화면에는 건수만 낸다.
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.seal import content_hash, outside_repo, read_key, read_manifest, set_hash  # noqa: E402

KIND = "pdf_hmac"


def pdf_hashes(pdf_dir, key):
    p = Path(pdf_dir)
    if not p.is_dir():
        sys.exit(f"[거부] PDF 폴더가 없다: {p}")
    files = sorted(p.glob("*.pdf"))
    if not files:
        sys.exit("[거부] PDF가 0개다")
    return {f.name: content_hash(f.read_bytes(), key) for f in files}


def write(pdf_dir, key, manifest):
    manifest = Path(manifest)
    if read_manifest(manifest):
        sys.exit(f"[거부] {manifest}에 이미 봉인이 있다. 봉인은 1회만 한다")
    hashes = pdf_hashes(pdf_dir, key)
    if len(set(hashes.values())) != len(hashes):
        sys.exit("[거부] 내용이 같은 PDF가 있다")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["kind", "sha256"])
        w.writerows([(KIND, h) for h in sorted(hashes.values())] + [(f"{KIND}_set", set_hash(hashes))])
    print(f"봉인: PDF {len(hashes)}건 → {manifest} (파일 이름 없음)")


def verify(pdf_dir, key, manifest):
    rows = read_manifest(manifest)
    sealed = Counter(h for k, h in rows if k == KIND)
    sealed_set = [h for k, h in rows if k == f"{KIND}_set"]
    if not sealed or len(sealed_set) != 1:
        return False, "봉인 기록 없음"
    hashes = pdf_hashes(pdf_dir, key)
    now = Counter(hashes.values())
    extra, lacking = now - sealed, sealed - now
    if extra or lacking:
        return False, f"봉인과 다른 PDF {sum(extra.values())}개, 봉인에 있는데 없는 PDF {sum(lacking.values())}개"
    if set_hash(hashes) != sealed_set[0]:
        return False, "내용은 같지만 파일 이름이 봉인과 다르다"
    return True, f"PDF {len(hashes)}건 일치"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--verify", action="store_true")
    ap.add_argument("--pdf-dir", required=True, help="저장소 밖 PDF 폴더")
    ap.add_argument("--key-file", required=True, help="저장소 밖 비밀 키 파일")
    ap.add_argument("--manifest", required=True, help="봉인 기록 파일(저장소 안, 커밋)")
    args = ap.parse_args(argv)
    if not outside_repo(args.pdf_dir):
        sys.exit("[거부] 확인용 PDF는 저장소 밖에 둔다")
    key = read_key(args.key_file)
    if args.write:
        write(args.pdf_dir, key, args.manifest)
        return
    ok, msg = verify(args.pdf_dir, key, args.manifest)
    print(("일치: " if ok else "불일치: ") + msg)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
