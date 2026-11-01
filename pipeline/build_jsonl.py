"""
[강덕우] train.jsonl / val.jsonl 생성 + 생성 보고서

  python3 pipeline/build_jsonl.py --dry-run          # 건수 · 유형별 수 · 거부 목록 · 길이만 출력(파일 안 씀)
  python3 pipeline/build_jsonl.py                    # data/train.jsonl, data/val.jsonl, data/build_report.json
  python3 pipeline/build_jsonl.py --no-tokens ...    # 토크나이저 없이(길이 측정 생략) — 코드 시험용
  python3 pipeline/build_jsonl.py --prompt v2 ...    # data/prompt_v2/{train,val}.jsonl · build_report.json (v1 파일은 그대로)

--prompt(core/prompt.py PROMPTS, 기본 v1): 시스템 프롬프트 버전. v1이 아니면 --out-dir · --report 아래 prompt_<버전>/에 쓴다.
증강(난수 · 변형)은 프롬프트와 무관하므로 같은 옵션이면 버전끼리 system 메시지만 다르다.

최종 학습 세트는 tag label-rules-frozen · split-frozen 뒤에만 만든다. 그 전에는 test/fixtures/로 코드만 시험한다:
  python3 pipeline/build_jsonl.py --splits test/fixtures/splits.csv --label-dir test/fixtures/labels \
      --text-dir test/fixtures/text --out-dir /tmp/aug_check --dry-run

한 줄 = 한 학습 예시: {"doc_id", "split", "variant", "messages": [system, user, assistant]}
메시지는 core/prompt.py의 build_messages() · format_target()으로만 만든다 — 추론과 한 글자도 다르면 안 된다.

train (문서마다)
  orig          원본 그대로
  <렌더러> ×N   겉모양만 바꾼 변형 (pipeline/augment/renderers.py), 라벨 동일. 기본은 렌더러 전부
  secret ×k     성분 하나를 영업비밀로 가림, 라벨도 같이 바꿈 (mutators.py)
  nohcode ×1    H코드를 지움, 라벨 code → null
  ecnum ×1      --ecnum일 때만. CAS 뒤에 EC 번호 칸을 끼움, 라벨 동일
  <겨냥 렌더러> --targeted일 때만. renderers.TARGETED_RENDERERS 4종(값 공백 · 비해당 표현 · 구분 표기 · 줄 잇기), 라벨 동일.
                문서마다 따로 둔 난수를 써서, 켜도 나머지 예시는 끈 세트와 글자까지 같다(r1 → r1_aug는 예시 추가만)
val (문서마다)
  orig + 렌더러 전부. 내용 변형은 넣지 않는다 — val은 실제 분포를 봐야 한다.

건수는 목표가 아니라 결과다: 문서 수 × 변형 구성으로 나온 값을 그대로 쓰고, 맞추려고 변형을 늘리거나 문서를 빼지 않는다.
모든 변형은 check_forbidden.check() · 스키마 검사를 통과해야 들어가고, 거부된 것은 보고서에 남긴다.
"""
import argparse
import hashlib
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.prompt import (  # noqa: E402
    CHAT_TEMPLATE_KWARGS,
    DEFAULT_PROMPT,
    PROMPTS,
    build_messages,
    format_target,
    prompt_sha256,
    prompt_subdir,
)
from core.schema import check_schema  # noqa: E402
from pipeline.augment.check_forbidden import check  # noqa: E402
from pipeline.augment.mutators import MUTATORS  # noqa: E402
from pipeline.augment.renderers import RENDERERS, TARGETED_RENDERERS  # noqa: E402
from pipeline.guard import load_splits, refuse_sealed, require_splits  # noqa: E402

TRIES = 6
CODE_FILES = ["core/prompt.py", "core/schema.py", "src/schema.py", "pipeline/build_jsonl.py", "pipeline/guard.py",
              "pipeline/augment/renderers.py", "pipeline/augment/mutators.py", "pipeline/augment/check_forbidden.py"]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    return sha256_bytes(Path(p).read_bytes())


def example(doc_id, split, variant, text, label, prompt=DEFAULT_PROMPT):
    msgs = build_messages(text, prompt=prompt) + [{"role": "assistant", "content": format_target(label)}]
    return {"doc_id": doc_id, "split": split, "variant": variant, "messages": msgs}


def variants_for(doc_id, text, label, rng, split, args):
    """→ (예시 목록, 거부 목록, 해당 없음 목록)
    거부: 만들었지만 검사에 걸림. 해당 없음: 문서에 바꿀 곳이 없음(예: H코드 없는 문서의 nohcode)."""
    out, rejected, not_applicable = [example(doc_id, split, "orig", text, label, args.prompt)], [], []
    seen = {text}

    def problems(new_text, new_label, gone, mutation):
        probs = check(text, new_text, new_label, gone, content_mutation=mutation)
        probs += [f"스키마: {e}" for e in check_schema(new_label)[1]]
        if new_text in seen:
            probs.append("원본·다른 변형과 텍스트가 같음")
        return probs

    def add(name, make, mutation):
        """make() → (text, label, gone) | None. 난수가 원본과 같은 표기를 고를 수 있어 몇 번 다시 뽑는다."""
        probs = []
        for _ in range(TRIES):
            r = make()
            if r is None:
                not_applicable.append({"doc_id": doc_id, "variant": name})
                return
            probs = problems(*r, mutation)
            if not probs:
                seen.add(r[0])
                out.append(example(doc_id, split, name, r[0], r[1], args.prompt))
                return
        rejected.append({"doc_id": doc_id, "variant": name, "reasons": probs})

    for name, fn in RENDERERS.items():
        add(name, lambda fn=fn: (fn(text, rng), label, ()), False)
    if split != "train":
        return out, rejected, not_applicable
    for k in range(args.secret):
        add(f"secret{k + 1}", lambda: MUTATORS["secret"](text, label, rng), True)
    add("nohcode", lambda: MUTATORS["nohcode"](text, label, rng), True)
    if args.ecnum:
        add("ecnum", lambda: MUTATORS["ecnum"](text, label, rng), True)
    if args.targeted:
        trng = random.Random(f"{args.seed}-targeted-{doc_id}")  # split 난수를 건드리지 않는다
        for name, fn in TARGETED_RENDERERS.items():
            add(name, lambda fn=fn: (lambda t: None if t is None else (t, label, ()))(fn(text, trng)), False)
    if args.renderers < len(RENDERERS):
        # 렌더러 변형만 덜어낸다. 난수를 따로 써서, 나머지 예시(orig · 내용 변형)는 기본 세트와 글자까지 같게
        keep = set(random.Random(f"renderers-{doc_id}").sample(list(RENDERERS), args.renderers))
        out = [e for e in out if e["variant"] not in RENDERERS or e["variant"] in keep]
    return out, rejected, not_applicable


def label_flags(label):
    return {
        "영업비밀": any(c["is_substitute_data"] for c in label["ingredients"]),
        "H코드없음": any(h["code"] is None for h in label["hazard_statements"]),
    }


def token_lengths(rows, tokenizer_name):
    """입력 + 정답 전체 길이(학습 시퀀스 길이). 학습 · 추론과 같은 chat template을 쓴다."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    lens = []
    for r in rows:
        s = tok.apply_chat_template(r["messages"], tokenize=False, **CHAT_TEMPLATE_KWARGS)
        lens.append(len(tok(s, add_special_tokens=False)["input_ids"]))
    return lens


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))]


def build_split(split, splits, args):
    # split마다 난수를 따로 둔다 — train 옵션을 바꿔도 val.jsonl은 한 글자도 바뀌지 않게(r1 · r2 val loss 비교)
    rng = random.Random(f"{args.seed}-{split}")
    doc_ids = sorted(d for d, r in splits.items() if r["split"] == split)
    require_splits(doc_ids, {split}, splits, who=f"{split} 입력")
    rows, rejected, not_applicable, skipped, origs = [], [], [], [], {}
    for doc_id in doc_ids:
        tp, lp = Path(args.text_dir) / f"{doc_id}.txt", Path(args.label_dir) / f"{doc_id}.json"
        if not tp.exists() or not lp.exists():
            skipped.append({"doc_id": doc_id, "reason": "텍스트 없음" if not tp.exists() else "라벨 없음"})
            continue
        text = tp.read_text(encoding="utf-8")
        label = json.loads(lp.read_text(encoding="utf-8"))
        ok, errs = check_schema(label)
        if not ok:
            skipped.append({"doc_id": doc_id, "reason": f"원본 라벨 스키마 오류: {errs[0]}"})
            continue
        origs[doc_id] = label
        ex, rej, na = variants_for(doc_id, text, label, rng, split, args)
        rows += ex
        rejected += rej
        not_applicable += na
    rng.shuffle(rows)
    return rows, rejected, not_applicable, skipped, origs


def summarize(split, rows, rejected, not_applicable, skipped, origs, splits, args):
    n_docs = len(origs)
    kinds = Counter(r["variant"].rstrip("0123456789") for r in rows)
    by_form = defaultdict(lambda: {"docs": 0, "examples": 0})
    for d in origs:
        by_form[splits[d].get("form") or "?"]["docs"] += 1
    for r in rows:
        by_form[splits[r["doc_id"]].get("form") or "?"]["examples"] += 1
    ex_flags = Counter()
    for r in rows:
        for k, v in label_flags(json.loads(r["messages"][-1]["content"])).items():
            ex_flags[k] += v
    doc_flags = Counter()
    for label in origs.values():
        for k, v in label_flags(label).items():
            doc_flags[k] += v
    s = {
        "docs": n_docs,
        "examples": len(rows),
        "multiplier_incl_orig": round(len(rows) / max(n_docs, 1), 2),
        "augmented_only": len(rows) - kinds.get("orig", 0),
        "variants": dict(sorted(kinds.items())),
        "by_form": {k: dict(v) for k, v in sorted(by_form.items())},
        "docs_with": dict(doc_flags),          # 원본 문서 중 해당 유형이 있는 문서 수
        "examples_with": dict(ex_flags),       # 증강 뒤 해당 유형이 있는 예시 수
        "skipped": skipped,
        "rejected": rejected,
        "not_applicable": dict(Counter(n["variant"] for n in not_applicable)),
        "not_applicable_docs": not_applicable,
    }
    if not args.no_tokens and rows:
        lens = token_lengths(rows, args.tokenizer)
        over = sorted(({"doc_id": r["doc_id"], "variant": r["variant"], "tokens": n}
                       for r, n in zip(rows, lens) if n > args.max_length), key=lambda x: -x["tokens"])
        longest = max(zip(lens, rows), key=lambda x: x[0])
        s["tokens"] = {"max_length": args.max_length, "p50": pct(lens, 50), "p90": pct(lens, 90),
                       "p95": pct(lens, 95), "max": max(lens), "mean": round(statistics.mean(lens), 1),
                       "longest": f"{longest[1]['doc_id']}:{longest[1]['variant']}", "over_limit": over}
    return s


def print_summary(split, s):
    print(f"\n[{split}] 문서 {s['docs']}건 → 예시 {s['examples']}건 (원본 포함 {s['multiplier_incl_orig']}배, 증강분 {s['augmented_only']}건)")
    print("  변형: " + " · ".join(f"{k} {v}" for k, v in s["variants"].items()))
    print("  서식: " + " · ".join(f"{k} 문서 {v['docs']}/예시 {v['examples']}" for k, v in s["by_form"].items()))
    print(f"  유형(문서 → 예시): " + " · ".join(f"{k} {s['docs_with'].get(k, 0)} → {s['examples_with'].get(k, 0)}"
                                           for k in ("영업비밀", "H코드없음")))
    if "tokens" in s:
        t = s["tokens"]
        print(f"  길이(입력+정답 토큰): P50 {t['p50']} · P90 {t['p90']} · P95 {t['p95']} · MAX {t['max']} ({t['longest']})"
              f" · {t['max_length']} 초과 {len(t['over_limit'])}건")
        for o in t["over_limit"]:
            print(f"    [초과] {o['doc_id']} {o['variant']}: {o['tokens']}")
    if s["not_applicable"]:
        print("  해당 없음(바꿀 곳 없는 문서): " + " · ".join(f"{k} {v}" for k, v in s["not_applicable"].items()))
    for sk in s["skipped"]:
        print(f"  [제외] {sk['doc_id']}: {sk['reason']}")
    for rj in s["rejected"]:
        more = f" 외 {len(rj['reasons']) - 1}건" if len(rj["reasons"]) > 1 else ""
        print(f"  [거부] {rj['doc_id']} {rj['variant']}: {rj['reasons'][0]}{more}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--secret", type=int, default=2, help="train 문서당 영업비밀 변형 수")
    ap.add_argument("--renderers", type=int, default=len(RENDERERS), help="train 문서당 렌더러 변형 수(val은 늘 전부)")
    ap.add_argument("--ecnum", action="store_true", help="EC 칸 변형 켜기(기본 끔, mutators.add_ec_column 참고)")
    ap.add_argument("--targeted", action="store_true", help="오류 겨냥 렌더러 4종 켜기(기본 끔, train만, cv/prereg_r1_aug.md)")
    ap.add_argument("--splits", default=str(ROOT / "data" / "splits.csv"))
    ap.add_argument("--label-dir", default=str(ROOT / "data" / "labels"))
    ap.add_argument("--text-dir", default=str(ROOT / "data" / "text"))
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    ap.add_argument("--train-out", default="train.jsonl")
    ap.add_argument("--val-out", default="val.jsonl")
    ap.add_argument("--report", default=None, help="보고서 경로(기본: out-dir/build_report.json)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    ap.add_argument("--max-length", type=int, default=4096, help="이 길이를 넘는 예시를 목록으로 남긴다(버리지 않음)")
    ap.add_argument("--no-tokens", action="store_true", help="길이 측정 생략(코드 시험용)")
    ap.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 요약만 출력")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default=DEFAULT_PROMPT,
                    help=f"시스템 프롬프트 버전(기본 {DEFAULT_PROMPT}). v1이 아니면 out-dir/prompt_<버전>/에 쓴다")
    args = ap.parse_args()

    refuse_sealed(args.splits, args.label_dir, args.text_dir, args.out_dir)
    splits = load_splits(args.splits)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "args": vars(args),
        "inputs": {"splits_sha256": sha256_file(args.splits)},
        "code_sha256": {f: sha256_file(ROOT / f) for f in CODE_FILES if (ROOT / f).exists()},
        "prompt": {"version": args.prompt, "text_sha256": prompt_sha256(args.prompt)},
        "splits": {},
    }
    built = {}
    for split in ("train", "val"):
        rows, rejected, not_applicable, skipped, origs = build_split(split, splits, args)
        # 쓰기 직전에 한 번 더: 이 파일에 들어가는 문서는 모두 이 split이어야 한다
        require_splits({r["doc_id"] for r in rows}, {split}, splits, who=f"{split} JSONL")
        report["inputs"][f"{split}_labels_sha256"] = sha256_bytes(b"".join(
            (Path(args.label_dir) / f"{d}.json").read_bytes() for d in sorted(origs)))
        s = summarize(split, rows, rejected, not_applicable, skipped, origs, splits, args)
        print_summary(split, s)
        built[split] = rows
        report["splits"][split] = s

    overlap = sorted({r["doc_id"] for r in built["train"]} & {r["doc_id"] for r in built["val"]})
    report["checks"] = {"train_val_doc_overlap": overlap,
                        "val_variants_in_train": sum(r["split"] != "train" for r in built["train"])}
    if overlap or report["checks"]["val_variants_in_train"]:
        raise SystemExit(f"train · val 문서가 섞임: {overlap}")
    print(f"\n[검사] train · val 문서 겹침 0 · train JSONL 안의 val 예시 0")

    if args.dry_run:
        return
    sub = prompt_subdir(args.prompt)  # v1은 ""(기존 위치), v2는 prompt_v2/ — v1 학습셋을 덮어쓰지 않게
    out_dir = Path(args.out_dir) / sub
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, name in (("train", args.train_out), ("val", args.val_out)):
        path = out_dir / name
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in built[split]).encode("utf-8")
        path.write_bytes(data)
        report["splits"][split]["jsonl"] = {"path": str(path), "lines": len(built[split]), "sha256": sha256_bytes(data)}
        print(f"  → {path}  sha256 {sha256_bytes(data)[:16]}…")
    rp = Path(args.report).parent / sub / Path(args.report).name if args.report else out_dir / "build_report.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {rp}")


if __name__ == "__main__":
    main()
