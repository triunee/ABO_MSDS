"""
[양세윤] 짝 비교 — 같은 문서에서 두 조건의 승 · 패 · 무 + 제조사 그룹 단위 부트스트랩 95% 신뢰구간 (docs/plan.md 5 · 10절)

eval/score.py가 남긴 문서별 상세(_score_detail.jsonl)를 읽어, 문서마다 지표별로 맞음(1) / 틀림(0)을 만들고
차이(a − b)를 본다. 신뢰구간은 문서가 아니라 제조사 그룹을 복원추출한다(같은 제조사 문서끼리는 독립이 아니라서).

  python3 eval/paired.py --a qlora_r1 --b base_fs --split val
  python3 eval/paired.py --a qlora_r3 --prompt-a v2_1 --b qlora_r1 --split val
  python3 eval/paired.py --a qlora_final --b base_fs --split test --allow-test --out-root <저장소 밖> \
                         --groups-csv <저장소 밖>          # --first: 비교군마다 최초 채점 상세로

- 그룹: val · train은 data/splits.csv의 split_group, test는 test 담당이 넘긴 (doc_id, group) CSV(봉인 대조).
- 주지표 기본값은 문서 완전 정답(확장 기준, doc_exact_ext). --metric으로 바꾼다(착수 회의 확정 전 후보).
- 결론(10절): test에서 --a qlora_final --b base_fs일 때만 결론 문구를 고른다. 악화 불허 필드(파싱률 · 스키마 ·
  제품명 · CAS F1 · pair F1)는 서식(subset)마다 점수 파일에서 a < b인지 본다.
- 기록: report/paired.csv (a, b, split, source, subset, metric, ...). 같은 (a, b, split, source)는 덮어쓴다.
  test는 문서 ID를 화면에 내지 않고, 문서별 승패는 저장소 밖 <out-root>/_paired_{a}_vs_{b}.jsonl에만 쓴다.
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.prompt import DEFAULT_PROMPT, PROMPTS, prompt_subdir  # noqa: E402
from eval.score import (  # noqa: E402
    DETAIL_FILE,
    FIRST_DETAIL_FILE,
    FIRST_FILE,
    OUTPUT_ROOT,
    SPLITS,
    SPLITS_CSV,
    TEST_CONDITIONS,
    outside_repo,
    read_log,
    read_splits,
    scores_csv_for,
)
from eval.seal import require_sealed  # noqa: E402

PAIRED_CSV = ROOT / "report" / "paired.csv"
DEFAULT_METRIC = "doc_exact_ext"
N_BOOT = 10000
SEED = 42
# 10절 악화 불허 필드 — score.py 지표 이름
NO_WORSE = ("parse_rate", "schema_rate", "product_name_acc", "cas_f1", "pair_f1")
FIELDS = ("parse_rate", "schema_rate", "product_name_acc", "signal_word_acc", "ghs_classification",
          "hazard_statements", "cas", "pair", "nocas", "doc_exact", "doc_exact_ext")
COLUMNS = ["a", "b", "split", "source", "subset", "metric", "n_docs", "n_groups", "win", "loss", "tie",
           "mean_diff", "ci_low", "ci_high", "n_boot", "seed"]


# ---------------------------------------------------------------- 문서 1건 → 지표별 맞음/틀림
def doc_outcomes(d):
    """score.py 문서별 상세 1줄 → {지표: 0/1}. 리스트 필드는 목록 상태까지 맞아야 1."""
    parsed = "parse_error" not in d
    return {
        "parse_rate": int(parsed),
        "schema_rate": int(parsed and "schema_errors" not in d),
        "product_name_acc": int(d["product_name"]["ok"]),
        "signal_word_acc": int(d["signal_word"]["ok"]),
        "ghs_classification": int(d["ghs"] == "ok" and "ghs_status" not in d),
        "hazard_statements": int(d["hcode"] == "ok" and d["htext"] == "ok" and "hazard_status" not in d),
        "cas": int(d["cas"] == "ok"),
        "pair": int(d["pair"] == "ok"),
        "nocas": int(d["nocas"] == "ok"),
        "doc_exact": int(d["exact"]),
        "doc_exact_ext": int(d["exact_ext"]),
    }


def read_detail(path):
    """→ (머리줄, {doc_id: 상세})"""
    if not path.exists():
        sys.exit(f"[거부] 채점 상세가 없다: {path}. eval/score.py를 먼저 돌릴 것")
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return lines[0], {d["doc_id"]: d for d in lines[1:]}


# ---------------------------------------------------------------- 짝 비교 + 그룹 부트스트랩
def bootstrap_ci(diffs, group_of, n_boot=N_BOOT, seed=SEED):
    """그룹 복원추출로 평균 차이의 95% 백분위 구간. 그룹이 1개뿐이면 구할 수 없어 (None, None)."""
    by_group = defaultdict(list)
    for doc_id, x in diffs.items():
        by_group[group_of[doc_id]].append(x)
    groups = [(sum(v), len(v)) for _, v in sorted(by_group.items())]
    if len(groups) < 2:
        return None, None
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        s = n = 0
        for _ in groups:
            gs, gn = groups[rng.randrange(len(groups))]
            s += gs
            n += gn
        stats.append(s / n)
    stats.sort()
    return stats[int(0.025 * n_boot)], stats[min(n_boot - 1, int(0.975 * n_boot))]


def compare(out_a, out_b, group_of, doc_ids, n_boot=N_BOOT, seed=SEED):
    """doc_ids 안에서 지표마다 승 · 패 · 무, 평균 차이, 그룹 부트스트랩 구간 → [dict]"""
    rows = []
    for metric in FIELDS:
        diffs = {d: out_a[d][metric] - out_b[d][metric] for d in doc_ids}
        lo, hi = bootstrap_ci(diffs, group_of, n_boot, seed)
        rows.append({
            "metric": metric,
            "n_docs": len(diffs),
            "n_groups": len({group_of[d] for d in doc_ids}),
            "win": sum(x > 0 for x in diffs.values()),
            "loss": sum(x < 0 for x in diffs.values()),
            "tie": sum(x == 0 for x in diffs.values()),
            "mean_diff": mean(diffs.values()),
            "ci_low": lo,
            "ci_high": hi,
        })
    return rows


def conclusion(ci_low, ci_high, worsened, token_ratio, n_docs):
    """docs/plan.md 10절 결론 문구(qlora_final 대 base_fs, 주지표)."""
    tok = f"입력 토큰 비율(qlora_final / base_fs) {token_ratio:.2f}" if token_ratio is not None else "입력 토큰 비율 없음"
    if ci_low is None:
        return f"신뢰구간을 구할 수 없다(그룹 1개). 문서 수 · 그룹 수와 승패만 보고. {tok}"
    if ci_low > 0 and not worsened:
        return (f"학습의 이점이 확인됐다 — \"재시작 학습에 쓰지 않은 제조사 {n_docs}건에서 학습의 이점이 확인됐다"
                f"(파일럿 노출 이력 있음)\" (문서 수 · 그룹 수 · 신뢰구간 함께). {tok}")
    if ci_low > 0:
        return f"신뢰구간 하한 > 0이지만 악화 불허 필드가 떨어져 우위 결론을 내지 않는다: {worsened}. {tok}"
    if ci_high < 0:
        return f"\"few-shot보다 정확도가 낮다\" + 떨어진 필드, 입력 토큰 절감과의 교환 관계. {tok}"
    return f"\"few-shot과의 정확도 차이를 확인하지 못했다\"(동등하다는 뜻이 아니다). {tok}"


# ---------------------------------------------------------------- 점수 파일(악화 불허 필드) · 토큰
def load_metrics(condition, split, prompt, pred_dir, first):
    """→ {(subset, metric): 값}. --first면 최초 채점 기록, 아니면 report/scores*.csv."""
    out = {}
    if first:
        path = pred_dir / FIRST_FILE
        if not path.exists():
            sys.exit(f"[거부] 최초 채점 기록이 없다: {path}")
        for line in path.read_text(encoding="utf-8").splitlines()[1:]:
            r = json.loads(line)
            out[(r["subset"], r["metric"])] = r["value"]
        return out
    path = scores_csv_for(prompt)
    if path.exists():
        with open(path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if r["condition"] == condition and r["split"] == split and r["value"] != "":
                    out[(r["subset"], r["metric"])] = float(r["value"])
    return out


def worsened_fields(m_a, m_b):
    """악화 불허 필드 중 a < b인 (subset, metric). 한쪽에만 값이 있으면 판단할 수 없어 따로 알린다."""
    bad, unknown = [], []
    for key in sorted({k for k in m_a | m_b if k[1] in NO_WORSE}):
        if key not in m_a or key not in m_b or m_a[key] is None or m_b[key] is None:
            unknown.append(key)
        elif m_a[key] < m_b[key]:
            bad.append(key)
    return bad, unknown


def input_token_ratio(dir_a, dir_b):
    means = []
    for d in (dir_a, dir_b):
        toks = [r["input_tokens"] for r in read_log(d).values() if not r.get("skipped") and r.get("input_tokens")]
        means.append(mean(toks) if toks else None)
    return means[0] / means[1] if all(means) else None


def fmt(v):
    if v is None:
        return ""
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def upsert(rows, key, path=None):
    path = path or PAIRED_CSV
    kept = []
    if path.exists():
        with open(path, encoding="utf-8-sig", newline="") as f:
            kept = [r for r in csv.DictReader(f) if tuple(r[k] for k in ("a", "b", "split", "source")) != key]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(kept)
        w.writerows({k: fmt(r.get(k)) for k in COLUMNS} for r in rows)


# ---------------------------------------------------------------- 실행
def read_groups_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["doc_id"]: r["group"] for r in csv.DictReader(f)}


def gate_test(args):
    if not args.allow_test:
        sys.exit("[거부] test 짝 비교는 --allow-test가 있어야 한다(비교군 채점이 끝난 뒤)")
    for c in (args.a, args.b):
        if c not in TEST_CONDITIONS:
            sys.exit(f"[거부] test 비교군은 {TEST_CONDITIONS}뿐이다")
    for name in ("out_root", "groups_csv"):
        v = getattr(args, name)
        if not v or not outside_repo(v):
            sys.exit(f"[거부] test는 저장소 밖 --{name.replace('_', '-')}가 필요하다")
    require_sealed("group", args.groups_csv, args.manifest)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="비교할 조건(차이 = a − b)")
    ap.add_argument("--b", required=True, help="기준 조건")
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--prompt-a", choices=sorted(PROMPTS), default=DEFAULT_PROMPT)
    ap.add_argument("--prompt-b", choices=sorted(PROMPTS), default=DEFAULT_PROMPT)
    ap.add_argument("--metric", choices=FIELDS, default=DEFAULT_METRIC, help=f"주지표(기본 {DEFAULT_METRIC})")
    ap.add_argument("--first", action="store_true", help="test: 비교군마다 최초 채점 상세로 비교")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--splits", default=SPLITS_CSV, help="val · train 그룹(split_group)을 읽을 분할표")
    ap.add_argument("--groups-csv", help="test 전용: (doc_id, group) — test 담당이 넘김(봉인 대조)")
    ap.add_argument("--out-root", help="출력 루트(기본 outputs/, test는 저장소 밖 필수)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-write", action="store_true", help="report/paired.csv에 쓰지 않음")
    ap.add_argument("--manifest", help="test 봉인 기록(기본 report/test_manifest.csv = 팀 test)")
    args = ap.parse_args(argv)

    is_test = args.split == "test"
    if is_test:
        gate_test(args)
    elif args.first:
        sys.exit("--first는 test 전용이다(최초 채점 기록은 test에만 있음)")
    out_root = Path(args.out_root or OUTPUT_ROOT)
    dir_a = out_root / prompt_subdir(args.prompt_a) / args.a / args.split
    dir_b = out_root / prompt_subdir(args.prompt_b) / args.b / args.split
    detail_name = FIRST_DETAIL_FILE if args.first else DETAIL_FILE
    head_a, det_a = read_detail(dir_a / detail_name)
    head_b, det_b = read_detail(dir_b / detail_name)
    if set(det_a) != set(det_b):
        sys.exit(f"[거부] 두 조건의 채점 문서가 다르다({len(det_a)}건 / {len(det_b)}건). 같은 문서 목록으로 채점할 것")
    if head_a.get("include_non_ghs") != head_b.get("include_non_ghs"):
        sys.exit("[거부] 두 조건의 --include-non-ghs 설정이 다르다")

    group_of = (read_groups_csv(args.groups_csv) if is_test
                else {d: r["split_group"] for d, r in read_splits(args.splits).items()})
    no_group = [d for d in det_a if not group_of.get(d)]
    if no_group:
        sys.exit(f"[거부] 그룹이 없는 문서 {len(no_group)}건" + ("" if is_test else f": {no_group}"))

    out_a = {d: doc_outcomes(x) for d, x in det_a.items()}
    out_b = {d: doc_outcomes(x) for d, x in det_b.items()}
    subsets = defaultdict(list)
    for d, x in det_a.items():
        subsets[x["subset"]].append(d)
    rows = [{"subset": "all", **r} for r in compare(out_a, out_b, group_of, sorted(det_a), args.n_boot, args.seed)]
    if len(subsets) > 1:
        for s, ids in sorted(subsets.items()):
            rows += [{"subset": s, **r} for r in compare(out_a, out_b, group_of, sorted(ids), args.n_boot, args.seed)]
    source = "first" if args.first else "latest"
    for r in rows:
        r.update(a=args.a, b=args.b, split=args.split, source=source, n_boot=args.n_boot, seed=args.seed)

    # 화면: 지표별 승패 · 차이 · 구간 (subset별)
    print(f"\n[{args.a} − {args.b} / {args.split} / {source}]  "
          f"문서 {rows[0]['n_docs']}건 · 그룹 {rows[0]['n_groups']}개 · 부트스트랩 {args.n_boot}회(seed {args.seed})")
    print(f"{'subset':<14}{'metric':<22}{'win':>5}{'loss':>6}{'tie':>5}{'diff':>9}{'95% CI':>20}")
    for r in rows:
        ci = f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]" if r["ci_low"] is not None else "(그룹 1개)"
        star = " *" if r["metric"] == args.metric else ""
        print(f"{r['subset']:<14}{r['metric']:<22}{r['win']:>5}{r['loss']:>6}{r['tie']:>5}"
              f"{r['mean_diff']:>+9.4f}{ci:>20}{star}")
    print(f"(* 주지표 {args.metric}. 차이 = 문서별 맞음(1)/틀림(0)의 {args.a} − {args.b} 평균)")

    bad, unknown = worsened_fields(load_metrics(args.a, args.split, args.prompt_a, dir_a, args.first),
                                   load_metrics(args.b, args.split, args.prompt_b, dir_b, args.first))
    print(f"\n악화 불허 필드 {NO_WORSE}: " + (f"{args.a}가 낮음 {bad}" if bad else "떨어진 것 없음"))
    if unknown:
        print(f"  [경고] 점수가 한쪽에만 있거나 비어 판단 못 함: {unknown}")
    ratio = input_token_ratio(dir_a, dir_b)
    if ratio is not None:
        print(f"입력 토큰 평균 비율({args.a} / {args.b}): {ratio:.3f}")
    if is_test and (args.a, args.b) == ("qlora_final", "base_fs"):
        main_row = next(r for r in rows if r["subset"] == "all" and r["metric"] == args.metric)
        print("\n결론(docs/plan.md 10절): " + conclusion(main_row["ci_low"], main_row["ci_high"], bad, ratio,
                                                          main_row["n_docs"]))
    elif not is_test:
        print("\n(val은 조건 선택용 — 결론 문구 · 최종 수치로 쓰지 않는다)")

    if args.no_write:
        return
    upsert(rows, (args.a, args.b, args.split, source))
    print(f"\n기록: {PAIRED_CSV}")
    if is_test:  # 문서별 승패는 저장소 밖에만(실패 분석용, 평가 뒤에 봄)
        per_doc = out_root / f"_paired_{args.a}_vs_{args.b}_{source}.jsonl"
        with open(per_doc, "w", encoding="utf-8") as f:
            for d in sorted(det_a):
                f.write(json.dumps({"doc_id": d, "subset": det_a[d]["subset"], "group": group_of[d],
                                    "a": out_a[d], "b": out_b[d]}, ensure_ascii=False) + "\n")
        print(f"문서별 승패: {per_doc} (저장소 밖)")


if __name__ == "__main__":
    main()
