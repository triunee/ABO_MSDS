"""
[양세윤] 새 test 평가(cv/prereg_newtest.md) — 비교군 12개(base_zs · base_fs · r1 fold 어댑터 5 · r1_aug fold 어댑터 5)의
추론 · 채점을 돌리고, 사전 등록 4절의 판정을 계산한다.

<NEWTEST_ROOT>(저장소 밖, 정답 작업 쪽이 만들어 넘김)
  text/         1~3항 텍스트 *.txt (+ _cut_log.csv)
  labels/       정답 *.json
  subset.csv    doc_id,subset   (서식: 현행 · 수입품 국문판 · 구서식)
  groups.csv    doc_id,group    (제조사 그룹, 관계사는 같은 그룹)
  out/<단위>/   단위마다 출력 · 채점 상세(eval/infer.py · eval/score.py가 씀). 저장소에 넣지 않는다

  python3 eval/newtest.py status --root <NEWTEST_ROOT>            # 단위별 출력 수 · 채점 여부(건수만)
  python3 eval/newtest.py run    --root <NEWTEST_ROOT> [--dry-run] # 단위마다 infer → score. 끝난 단계는 건너뜀
  python3 eval/newtest.py report --root <NEWTEST_ROOT>            # 판정 → cv/newtest/report.md · scores.csv · paired.csv
  python3 eval/newtest.py verify-adapters                         # 어댑터 10개 폴더 해시 = experiment.json

- 고정: cv/newtest/experiment.json(어댑터 해시 · 생성 길이 · 프롬프트), 봉인: cv/newtest/test_manifest.csv.
  infer · score에 --experiment · --manifest로 넘겨 팀 test(eval/experiment.json · report/test_manifest.csv)와 섞지 않는다.
- 판정(사전 등록 4절, 주지표 doc_exact_ext, 제조사 그룹 부트스트랩 95% 구간):
  - 주 판정 r1 vs base_fs: 문서마다 (r1 fold 어댑터 5개 평균) − base_fs. 악화 불허 = parse_rate · schema_rate ·
    product_name_acc · cas_f1 · pair_f1(r1은 5개 평균)
  - 레시피 판정 r1_aug vs r1: 문서마다 5개 평균끼리. 악화 불허 = parse_rate · schema_rate · cas_f1 · pair_f1
- report는 단위마다 최초 채점(_score_first.jsonl)이 있어야 하고, 그 뒤 채점기 · 별칭표가 바뀌었으면 거부한다.
  저장소에는 문서 이름 없는 집계만 쓴다. 문서별 결과는 <NEWTEST_ROOT>/out/_newtest_per_doc.jsonl(저장소 밖)에만.
"""
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.prompt import DEFAULT_PROMPT  # noqa: E402
from eval.cv import REPORT_METRICS  # noqa: E402
from eval.infer import read_cut_status  # noqa: E402
from eval.paired import FIELDS, N_BOOT, SEED, compare, doc_outcomes, read_groups_csv  # noqa: E402
from eval.score import (  # noqa: E402
    ALIAS_CSV,
    FIRST_DETAIL_FILE,
    FIRST_FILE,
    SCORER_FILES,
    check_prompt_version,
    outside_repo,
    read_subset_csv,
    score_docs,
    sha256_files,
)
from eval.seal import KINDS, require_sealed  # noqa: E402

NT_DIR = ROOT / "cv" / "newtest"
EXPERIMENT = NT_DIR / "experiment.json"
MANIFEST = NT_DIR / "test_manifest.csv"
PY = sys.executable
METRIC = "doc_exact_ext"
GUARD_MAIN = ("parse_rate", "schema_rate", "product_name_acc", "cas_f1", "pair_f1")  # 4-1 악화 불허
GUARD_RECIPE = ("parse_rate", "schema_rate", "cas_f1", "pair_f1")  # 4-2 악화 불허
BASES = ("base_zs", "base_fs")


# ---------------------------------------------------------------- 설정 · 경로
def load_exp(path=None):
    exp = json.loads(Path(path or EXPERIMENT).read_text(encoding="utf-8"))
    for name, units in exp["recipes"].items():
        lacking = [u for u in units if u not in exp["adapters"] or u not in exp["adapter_paths"]]
        if lacking:
            sys.exit(f"[거부] {name}의 단위 {lacking}에 어댑터 해시 · 경로가 없다")
    return exp


def units(exp):
    """→ [(단위 이름, infer 조건, 어댑터 경로 | None)]. 어댑터 단위는 조건 이름 qlora_final, 출력 폴더는 단위마다."""
    out = [(b, b, None) for b in BASES]
    out += [(u, "qlora_final", ROOT / exp["adapter_paths"][u]) for r in exp["recipes"].values() for u in r]
    return out


def paths(root):
    root = Path(root)
    if not outside_repo(root):
        sys.exit("[거부] 새 test 자료(--root)는 저장소 밖에 둔다")
    return {"label": root / "labels", "text": root / "text", "subset": root / "subset.csv",
            "group": root / "groups.csv", "out": root / "out"}


def pred_dir(p, unit, cond):
    return p["out"] / unit / cond / "test"


def doc_list(text_dir):
    """infer.py의 test 문서 목록과 같은 규칙: *.txt + _cut_log에만 있는 실패 문서"""
    return sorted({f.stem for f in Path(text_dir).glob("*.txt")} | set(read_cut_status(text_dir)))


def n_outputs(d):
    return len(list(d.glob("*.json"))) if d.exists() else 0


# ---------------------------------------------------------------- status · run
def cmd_status(args):
    exp, p = load_exp(args.experiment), paths(args.root)
    total = len(doc_list(p["text"]))
    print(f"문서 {total}건 (이름은 내지 않음)")
    print(f"{'단위':<12}{'출력':>8}{'채점':>6}")
    for unit, cond, _ in units(exp):
        d = pred_dir(p, unit, cond)
        print(f"{unit:<12}{n_outputs(d):>5}/{total:<3}{'예' if (d / FIRST_FILE).exists() else '-':>5}")


def cmd_verify_adapters(args):
    """단위마다 어댑터 폴더 해시 = experiment.json의 그 단위 해시(다른 fold 어댑터가 자리를 바꿔 들어간 것까지 잡는다)"""
    from eval.infer import sha256_dir
    exp = load_exp(args.experiment)
    bad = []
    for unit, _, adapter in units(exp):
        if adapter is None:
            continue
        got = sha256_dir(adapter) if adapter.is_dir() else "(폴더 없음)"
        ok = got == exp["adapters"][unit]
        print(f"{unit:<12}{'일치' if ok else '불일치'}  {got[:12]}")
        bad += [] if ok else [unit]
    if bad:
        sys.exit(f"[거부] 어댑터 해시가 고정 파일과 다르다: {bad}")
    print("어댑터 10개 모두 일치" if len(exp["adapters"]) == 10 else f"어댑터 {len(exp['adapters'])}개 모두 일치")


def commands(exp, p, unit, cond, adapter, experiment, manifest):
    common = ["--split", "test", "--allow-test", "--out-root", str(p["out"] / unit), "--text-dir", str(p["text"]),
              "--experiment", str(experiment), "--manifest", str(manifest)]
    infer = [PY, "eval/infer.py", "--condition", cond, *common] + (["--adapter", str(adapter)] if adapter else [])
    score = [PY, "eval/score.py", "--condition", cond, *common, "--label-dir", str(p["label"]),
             "--subset-csv", str(p["subset"])]
    return infer, score


def cmd_run(args):
    exp, p = load_exp(args.experiment), paths(args.root)
    experiment, manifest = Path(args.experiment or EXPERIMENT), Path(args.manifest or MANIFEST)
    if not args.dry_run:
        for kind in KINDS:  # 정답 · 서식 · 그룹까지 봉인된 뒤에만 첫 추론을 건다(사전 등록 8절 순서)
            require_sealed(kind, p[kind], manifest)
    total = len(doc_list(p["text"]))
    todo = [u for u in units(exp) if not (pred_dir(p, u[0], u[1]) / FIRST_FILE).exists()]
    print(f"단위 {len(units(exp))}개 중 남은 것 {len(todo)}개, 문서 {total}건")
    for unit, cond, adapter in todo:
        infer, score = commands(exp, p, unit, cond, adapter, experiment, manifest)
        steps = ([] if n_outputs(pred_dir(p, unit, cond)) >= total else [("infer", infer)]) + [("score", score)]
        for name, cmd in steps:
            print(f"\n[{unit}] {name}: " + " ".join(cmd))
            if args.dry_run:
                continue
            r = subprocess.run(cmd, cwd=ROOT)
            if r.returncode != 0:
                sys.exit(f"[멈춤] {unit} {name} 실패(종료 코드 {r.returncode}). 원인을 확인한 뒤 같은 명령으로 이어 돌린다")


# ---------------------------------------------------------------- report
def mean_or_none(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def avg_outcomes(outs, names, ids):
    return {d: {f: sum(outs[n][d][f] for n in names) / len(names) for f in FIELDS} for d in ids}


def first_meta(d):
    lines = (d / FIRST_FILE).read_text(encoding="utf-8").splitlines()
    return json.loads(lines[0])


def check_first(d, details):
    """최초 채점 뒤 채점기 · 별칭표가 바뀌지 않았고, 지금 채점이 최초 채점과 문서마다 같은지."""
    meta = first_meta(d)
    if meta.get("scorer_sha256") != sha256_files(SCORER_FILES) or meta.get("alias_sha256") != sha256_files([ALIAS_CSV]):
        sys.exit(f"[거부] {d.parts[-3]}: 최초 채점 뒤 채점기 · 별칭표가 바뀌었다. 판정은 최초 채점 기준으로만 낸다")
    first = {}
    for line in (d / FIRST_DETAIL_FILE).read_text(encoding="utf-8").splitlines()[1:]:
        x = json.loads(line)
        first[x["doc_id"]] = x["exact_ext"]
    if first != {k: v["exact_ext"] for k, v in details.items()}:
        sys.exit(f"[거부] {d.parts[-3]}: 지금 채점이 최초 채점과 다르다")


def score_unit(d, ids, subset_of, p):
    check_prompt_version(d, DEFAULT_PROMPT)
    acc_all, _, details = score_docs(ids, {i: "all" for i in ids}, p["label"], d, p["text"], quiet=True)
    acc_sub, _, _ = score_docs(ids, subset_of, p["label"], d, p["text"], quiet=True)
    check_first(d, details)
    metrics = {"all": dict(acc_all["all"].rows())} | {s: dict(a.rows()) for s, a in acc_sub.items()}
    return metrics, {i: doc_outcomes(details[i]) for i in ids}


def guard(m_a, m_b, names):
    """→ {지표: (a, b, a >= b)}. 값이 없으면 판단할 수 없어 실패로 둔다."""
    out = {}
    for m in names:
        a, b = m_a.get(m), m_b.get(m)
        out[m] = (a, b, a is not None and b is not None and a >= b)
    return out


def verdict_main(row, g, ratio, n_docs, n_groups):
    worse = [m for m, (_, _, ok) in g.items() if not ok]
    tok = f"입력 토큰 비율(r1 / base_fs) {ratio:.2f}" if ratio is not None else "입력 토큰 비율 없음"
    if row["ci_low"] is None:
        return f"구간을 구할 수 없다(제조사 그룹 1개). 문서 수 · 승패만 보고한다. {tok}"
    if row["ci_low"] > 0 and not worse:
        return f"학습에 쓰지 않은 제조사 {n_groups}곳 {n_docs}건에서 학습의 이점이 확인됐다. {tok}"
    if row["ci_low"] > 0:
        return f"구간 하한 > 0이지만 악화 불허 지표가 떨어져 \"이점이 확인됐다\"고 쓰지 않는다: {worse}. {tok}"
    if row["ci_high"] < 0:
        return f"few-shot보다 정확도가 낮다. 떨어진 악화 불허 지표: {worse or '없음'}. 입력 토큰 절감과의 교환 관계: {tok}"
    return f"few-shot과의 정확도 차이를 확인하지 못했다(동등하다는 뜻이 아니다). {tok}"


def verdict_recipe(row, g):
    failed = ([] if row["ci_low"] is not None and row["ci_low"] > 0 else ["구간 하한이 0 이하"])
    failed += [f"{m} 악화" for m, (_, _, ok) in g.items() if not ok]
    return ("새 문서에서 개선 확인" if not failed else "새 문서에서 개선을 확인하지 못했다"), failed


def fmt(v, metric=""):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.1f}" if metric.endswith(("_mean", "_s")) else f"{v:.3f}"
    return str(v)


def ci_text(r):
    return f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]" if r["ci_low"] is not None else "(그룹 1개)"


def cmd_report(args):
    exp, p = load_exp(args.experiment), paths(args.root)
    manifest = Path(args.manifest or MANIFEST)
    for kind in KINDS:
        require_sealed(kind, p[kind], manifest)
    subset_of, group_of = read_subset_csv(p["subset"]), read_groups_csv(p["group"])
    ids = sorted(subset_of)
    if set(group_of) != set(ids):
        sys.exit("[거부] 서식 목록과 그룹 목록의 문서가 다르다")
    missing = [u for u, c, _ in units(exp) if not (pred_dir(p, u, c) / FIRST_FILE).exists()]
    if missing:
        sys.exit(f"[거부] 채점이 끝나지 않은 단위: {missing}. eval/newtest.py run을 먼저 끝낼 것")

    metrics, outs = {}, {}
    for unit, cond, _ in units(exp):
        metrics[unit], outs[unit] = score_unit(pred_dir(p, unit, cond), ids, subset_of, p)
    recipes = exp["recipes"]
    subsets = ["all"] + sorted(set(subset_of.values()))
    # 조건별 지표: base는 그대로, 레시피는 단위 평균
    cond_metrics = {b: metrics[b] for b in BASES}
    for r, us in recipes.items():
        cond_metrics[r] = {s: {m: mean_or_none([metrics[u][s].get(m) for u in us]) for m in metrics[us[0]][s]}
                           for s in subsets}
    cond_outs = {b: outs[b] for b in BASES} | {r: avg_outcomes(outs, us, ids) for r, us in recipes.items()}

    def comp(a, b):
        rows = [{"subset": "all", **r} for r in compare(cond_outs[a], cond_outs[b], group_of, ids, args.n_boot, SEED)]
        for s in subsets[1:]:
            sid = [i for i in ids if subset_of[i] == s]
            rows += [{"subset": s, **r} for r in compare(cond_outs[a], cond_outs[b], group_of, sid, args.n_boot, SEED)]
        return [{"a": a, "b": b, **r} for r in rows]

    pairs = {k: comp(*k) for k in (("r1", "base_fs"), ("r1_aug", "r1"), ("r1", "base_zs"), ("base_fs", "base_zs"))}
    main_of = {k: next(r for r in rows if r["subset"] == "all" and r["metric"] == METRIC) for k, rows in pairs.items()}
    n_docs, n_groups = len(ids), len(set(group_of.values()))

    g_main = guard(cond_metrics["r1"]["all"], cond_metrics["base_fs"]["all"], GUARD_MAIN)
    tok_r1, tok_fs = (cond_metrics[c]["all"].get("input_tokens_mean") for c in ("r1", "base_fs"))
    ratio = tok_r1 / tok_fs if tok_r1 and tok_fs else None
    v_main = verdict_main(main_of[("r1", "base_fs")], g_main, ratio, n_docs, n_groups)
    g_rec = guard(cond_metrics["r1_aug"]["all"], cond_metrics["r1"]["all"], GUARD_RECIPE)
    v_rec, failed_rec = verdict_recipe(main_of[("r1_aug", "r1")], g_rec)
    flips = {r: sum(len({outs[u][i][METRIC] for u in us}) > 1 for i in ids) for r, us in recipes.items()}
    cut = read_cut_status(p["text"])
    n_fail = sum(cut.get(i) == "NOT_FOUND" for i in ids)

    NT_DIR.mkdir(parents=True, exist_ok=True)
    write_scores(cond_metrics | {u: metrics[u] for us in recipes.values() for u in us}, subsets)
    write_paired([r for rows in pairs.values() for r in rows], args.n_boot)
    report = render(exp, subset_of, n_docs, n_groups, n_fail, cond_metrics, subsets, pairs, main_of,
                    (v_main, g_main), (v_rec, failed_rec, g_rec), flips, args.n_boot)
    (NT_DIR / "report.md").write_text(report, encoding="utf-8")
    with open(p["out"] / "_newtest_per_doc.jsonl", "w", encoding="utf-8") as f:  # 저장소 밖(실패 분석용)
        for i in ids:
            f.write(json.dumps({"doc_id": i, "subset": subset_of[i], "group": group_of[i],
                                **{u: outs[u][i][METRIC] for u in outs}}, ensure_ascii=False) + "\n")
    print(f"주 판정(r1 vs base_fs): {v_main}")
    print(f"레시피 판정(r1_aug vs r1): {v_rec}" + (f" — {failed_rec}" if failed_rec else ""))
    print(f"기록: {NT_DIR / 'report.md'} · scores.csv · paired.csv (문서 이름 없음)")


def write_scores(cond_metrics, subsets):
    with open(NT_DIR / "scores.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["condition", "subset", "metric", "value"])
        for c, by_sub in cond_metrics.items():
            for s in subsets:
                for m, v in by_sub[s].items():
                    w.writerow([c, s, m, "" if v is None else (f"{v:.6f}" if isinstance(v, float) else v)])


def write_paired(rows, n_boot):
    cols = ["a", "b", "subset", "metric", "n_docs", "n_groups", "win", "loss", "tie", "mean_diff", "ci_low", "ci_high"]
    with open(NT_DIR / "paired.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["n_boot", "seed"], lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**{k: ("" if r[k] is None else (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]))
                           for k in cols}, "n_boot": n_boot, "seed": SEED})


def prereg_limits(exp):
    """사전 등록 파일의 '## 7. 한계' 절 본문(다음 '## ' 전까지)을 그대로 옮긴다"""
    lines, out, on = (ROOT / exp["prereg"]).read_text(encoding="utf-8").splitlines(), [], False
    for line in lines:
        if line.startswith("## "):
            on = line.startswith("## 7.")
            continue
        if on and line.strip():
            out.append(line)
    if not out:
        sys.exit(f"[거부] {exp['prereg']}에서 '## 7.' 한계 절을 찾지 못했다")
    return out


def render(exp, subset_of, n_docs, n_groups, n_fail, cm, subsets, pairs, main_of, vm, vr, flips, n_boot):
    from collections import Counter
    forms = Counter(subset_of.values())
    L = ["# 새 test 평가 결과", "",
         f"- 사전 등록: `{exp['prereg']}`(커밋 `{exp['prereg_commit']}`). 기준은 결과 전에 고정했고 이 보고서는 그대로 계산한 것이다.",
         f"- 문서 {n_docs}건 · 제조사 그룹 {n_groups}개 · 서식별 " + " · ".join(f"{k} {v}" for k, v in sorted(forms.items())),
         f"- 추출 실패(1~3항을 못 자름) {n_fail}건 — 분모에서 빼지 않음",
         f"- 주지표 {METRIC}, 제조사 그룹 부트스트랩 {n_boot}회(seed {SEED}) 95% 구간. r1 · r1_aug는 fold 어댑터 5개 평균",
         f"- 생성 {exp['gpu']} · 프롬프트 {exp['prompt_version']} · max_new_tokens {exp['max_new_tokens']}", "",
         "## 주 판정: r1 vs base_fs (사전 등록 4-1)", ""]
    r = main_of[("r1", "base_fs")]
    L += [f"- 차이 {r['mean_diff']:+.3f} {ci_text(r)}, 승 {r['win']} · 패 {r['loss']} · 무 {r['tie']}",
          f"- **결론: {vm[0]}**", "", "| 악화 불허 지표 | r1 | base_fs | 통과 |", "|---|---|---|---|"]
    L += [f"| {m} | {fmt(a)} | {fmt(b)} | {'예' if ok else '아니오'} |" for m, (a, b, ok) in vm[1].items()]
    r = main_of[("r1_aug", "r1")]
    L += ["", "## 레시피 판정: r1_aug vs r1 (사전 등록 4-2)", "",
          f"- 차이 {r['mean_diff']:+.3f} {ci_text(r)}, 승 {r['win']} · 패 {r['loss']} · 무 {r['tie']}",
          f"- **결론: {vr[0]}**" + (f" (사유: {', '.join(vr[1])})" if vr[1] else ""),
          "- CV 판정(`cv/prereg_r1_aug.md`)과 따로 보고한다. 둘을 합쳐 하나의 성공으로 만들지 않는다.", "",
          "| 악화 불허 지표 | r1_aug | r1 | 통과 |", "|---|---|---|---|"]
    L += [f"| {m} | {fmt(a)} | {fmt(b)} | {'예' if ok else '아니오'} |" for m, (a, b, ok) in vr[2].items()]
    L += ["", "## 참고 (판정에 쓰지 않음)", "", "### 짝 비교 (주지표, 전체)", "",
          "| a − b | 차이 | 95% 구간 | 승 | 패 | 무 |", "|---|---|---|---|---|---|"]
    for (a, b), m in main_of.items():
        L.append(f"| {a} − {b} | {m['mean_diff']:+.3f} | {ci_text(m)} | {m['win']} | {m['loss']} | {m['tie']} |")
    L += ["", "- fold 어댑터 5개 사이에서 주지표가 갈린 문서: " + " · ".join(f"{k} {v}건" for k, v in flips.items()), ""]
    for s in subsets:
        n = cm["base_fs"][s].get("n_docs")
        L += [f"### 지표 — {'전체' if s == 'all' else s} ({n}건)", "",
              "| 지표 | " + " | ".join(cm) + " |", "|---|" + "---|" * len(cm)]
        L += [f"| {m} | " + " | ".join(fmt(cm[c][s].get(m), m) for c in cm) + " |" for m in REPORT_METRICS]
        L.append("")
    L += ["## 사전 등록과 다른 점", ""]
    L += [f"- {d}" for d in exp.get("deviations", [])] or ["- 없음"]
    L += ["", "## 한계 (사전 등록 7절 그대로)", "", *prereg_limits(exp), ""]
    L.append(f"생성 {datetime.now().isoformat(timespec='seconds')} · `python3 eval/newtest.py report`")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", help="고정 파일(기본 cv/newtest/experiment.json)")
    ap.add_argument("--manifest", help="봉인 기록(기본 cv/newtest/test_manifest.csv)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify-adapters", help="어댑터 10개의 폴더 해시를 experiment.json과 대조")
    for name in ("status", "run", "report"):
        s = sub.add_parser(name)
        s.add_argument("--root", required=True, help="<NEWTEST_ROOT>(저장소 밖)")
        if name == "run":
            s.add_argument("--dry-run", action="store_true", help="명령만 출력(봉인 확인도 건너뜀)")
        if name == "report":
            s.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args(argv)
    {"status": cmd_status, "run": cmd_run, "report": cmd_report,
     "verify-adapters": cmd_verify_adapters}[args.cmd](args)


if __name__ == "__main__":
    main()
