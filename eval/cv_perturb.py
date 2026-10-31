"""
보지 않은 서식 변형 강건성 — 학습 증강(pipeline/augment/renderers.py 6종)에 없는 변형으로 평가 문서의 모양만 바꿔,
교차검증과 같은 fold 모델 · few-shot 예시로 다시 추론하고 원본 대비 하락 폭을 잰다.

  python3 eval/cv_perturb.py make                  # 변형 텍스트 생성 + 내용 보존 검사 → cv/perturb/text/
  python3 eval/cv_perturb.py run [--set layout]    # fold별 추론(base_fs · r1), 이미 있는 출력은 건너뜀
  python3 eval/cv_perturb.py report                # → cv/perturb/report.md · scores.csv

변형 세트(결과를 보기 전에 고정, 바꾸면 새 세트 이름으로 추가):
- layout — PDF 레이아웃 깨짐. 줄마다 seed로 정함
  - split_kv: "라벨 : 값" 줄의 절반을 "라벨 :" / "값" 두 줄로 나눔(콜론 뒤 공백 → 줄바꿈)
  - wrap: 30자를 넘는 줄을 30자 안쪽의 마지막 공백에서 줄바꿈(반복)
- value — 값 안 공백
  - CAS 번호 하이픈 양옆에 공백: 67-64-1 → 67 - 64 - 1
  - 함유량 범위 구분자(~ ∼ ～) 양옆에 공백: 10~20 → 10 ~ 20
  - 신호어 줄(같은 줄 또는 다음 줄)의 위험 · 경고를 위 험 · 경 고로

- 두 세트 모두 공백 · 줄바꿈만 넣거나 바꾼다. 공백을 모두 지운 문자열이 원본과 같아야 하고(make가 검사),
  그래서 정답 라벨을 그대로 쓴다. 정답 규칙도 이 변형에 영향받지 않는다(CAS · 함유량은 공백 제거, 신호어 변형은 위험 · 경고).
- 평가 문서만 바꾼다. few-shot 예시는 원본 그대로 둔다(fold마다 텍스트 폴더를 따로 만드는 이유).
- 채점은 원본 텍스트(data/text/)로 한다 — 무근거 생성 검사가 원문 문자열을 찾기 때문.
- 판정(사전 등록): 주지표 doc_exact_ext의 하락 폭(원본 − 변형)을 조건마다 제조사 그룹 부트스트랩 95% 구간으로 내고,
  r1 하락 − base_fs 하락의 구간 상한이 0보다 작으면 "r1이 이 변형에 더 강건하다", 0을 포함하면 "차이를 확인하지 못했다".
"""
import argparse
import fcntl
import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import cv  # noqa: E402
from eval.paired import bootstrap_ci, compare, doc_outcomes  # noqa: E402
from eval.score import score_docs  # noqa: E402

PERTURB_ROOT = cv.CV_ROOT / "perturb"
CONDITIONS = ("base_fs", "r1")  # cv/experiments.yaml의 조건 이름
SEED = 42
WRAP = 30

_KV = re.compile(r"^(.*?\S)(\s*[:：])[ \t]+(\S.*)$")
_CAS = re.compile(r"(?<![\d-])(\d{2,7})-(\d{2})-(\d)(?![\d-])")
_RANGE = re.compile(r"(\d)[ \t]*([~∼～])[ \t]*(\d)")
_SIGNAL = re.compile(r"(?<![가-힣])(위험|경고)(?![가-힣])")


def wrap_line(ln, width=WRAP):
    out = []
    while len(ln) > width:
        cut = ln.rfind(" ", 1, width + 1)
        if cut <= 0:
            break
        out.append(ln[:cut])
        ln = ln[cut + 1:]
    return out + [ln]


def layout(text, rng):
    lines = []
    for ln in text.split("\n"):
        m = _KV.match(ln)
        # 괄호 안 콜론("급성 독성(흡입: 가스)")은 값의 일부라 나누지 않는다
        if m and m.group(1).count("(") <= m.group(1).count(")") and rng.random() < 0.5:
            lines += [m.group(1) + m.group(2), m.group(3)]
        else:
            lines.append(ln)
    return "\n".join(w for ln in lines for w in wrap_line(ln))


def value(text, rng):
    text = _CAS.sub(r"\1 - \2 - \3", text)
    text = _RANGE.sub(r"\1 \2 \3", text)
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if "신호어" not in ln:
            continue
        head, _, tail = ln.partition("신호어")
        if _SIGNAL.search(tail):
            lines[i] = head + "신호어" + _SIGNAL.sub(lambda m: " ".join(m.group(1)), tail)
        elif i + 1 < len(lines):
            lines[i + 1] = _SIGNAL.sub(lambda m: " ".join(m.group(1)), lines[i + 1])
    return "\n".join(lines)


SETS = {"layout": layout, "value": value}


def squash_all(s):
    return re.sub(r"\s+", "", s)


def perturb(name, doc_id, text):
    """세트 · 문서마다 고정 seed. 공백을 지운 내용이 원본과 다르면 멈춘다."""
    out = SETS[name](text, random.Random(f"{SEED}:{name}:{doc_id}"))
    if squash_all(out) != squash_all(text):
        sys.exit(f"[거부] {name} 변형이 {doc_id}의 내용을 바꿨다(공백 밖 글자가 다름)")
    return out


def text_dir(name, k):
    return PERTURB_ROOT / "text" / name / f"fold{k}"


def out_root(name, cond, k):
    return PERTURB_ROOT / "outputs" / name / cond / f"fold{k}"


def cmd_make(args):
    fold_of = cv.read_folds(cv.CV_ROOT)
    stats = []
    for name in SETS:
        changed = total = docs_changed = 0
        for k in sorted(set(fold_of.values())):
            dst = text_dir(name, k)
            if dst.exists():
                shutil.rmtree(dst)
            dst.mkdir(parents=True)
            shutil.copyfile(cv.TEXT_DIR / "_cut_log.csv", dst / "_cut_log.csv")
            for d in sorted(fold_of):  # 평가 문서만 변형, 나머지(few-shot 후보)는 원본 복사
                src = (cv.TEXT_DIR / f"{d}.txt").read_text(encoding="utf-8")
                if fold_of[d] != k:
                    (dst / f"{d}.txt").write_text(src, encoding="utf-8")
                    continue
                out = perturb(name, d, src)
                (dst / f"{d}.txt").write_text(out, encoding="utf-8")
                a, b = src.split("\n"), set(out.split("\n"))
                changed += sum(ln not in b for ln in a)
                total += len(a)
                docs_changed += out != src
        stats.append(f"- {name}: 바뀐 줄 {changed}/{total} ({changed / total:.1%}), 바뀐 문서 {docs_changed}/{len(fold_of)}")
    print("\n".join(stats))
    (PERTURB_ROOT / "make_stats.md").write_text("\n".join(stats) + "\n", encoding="utf-8")


def cmd_run(args):
    exp = cv.load_experiments(cv.EXPERIMENTS)
    fold_of = cv.read_folds(cv.CV_ROOT)
    names = [args.set] if args.set else list(SETS)
    PERTURB_ROOT.mkdir(parents=True, exist_ok=True)
    with open(PERTURB_ROOT / ".lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit("[거부] 다른 cv_perturb run이 돌고 있다")
        for name in names:
            for cond in CONDITIONS:
                ctype = exp["conditions"][cond]["type"]
                for k in sorted(set(fold_of.values())):
                    p = cv.Paths(cv.CV_ROOT, cond, k)
                    if not text_dir(name, k).exists():
                        sys.exit("[거부] 변형 텍스트가 없다 — make 먼저")
                    cmd = [cv.PY, "eval/infer.py", "--condition", cv.INFER_CONDITION[ctype], "--split", "val",
                           "--splits", p.splits, "--text-dir", text_dir(name, k), "--out-root", out_root(name, cond, k)]
                    if ctype == "qlora":
                        cmd += ["--adapter", p.adapter]
                    if ctype == "base_fs":
                        cmd += ["--fewshot-json", p.fewshot]
                    print(f"\n[{name} / {cond} / fold {k}]")
                    rc = subprocess.run([str(c) for c in cmd], cwd=ROOT).returncode
                    if rc:
                        sys.exit(f"[중단] 종료 코드 {rc} — 같은 명령으로 이어서 돌린다")


def pool_perturbed(name, cond, ctype, fold_of):
    """fold별 변형 출력 → cv/perturb/pooled/{name}/{cond}/<infer 조건>/val (cv.pool과 같은 모양)"""
    dst = PERTURB_ROOT / "pooled" / name / cond / cv.INFER_CONDITION[ctype] / "val"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    logs, runs = [], []
    for k in sorted(set(fold_of.values())):
        src = out_root(name, cond, k) / cv.prompt_subdir(cv.DEFAULT_PROMPT) / cv.INFER_CONDITION[ctype] / "val"
        for d in cv.held_out(fold_of, k):
            if not (src / f"{d}.json").exists():
                return None
            shutil.copyfile(src / f"{d}.json", dst / f"{d}.json")
        logs += [x for x in (src / "_log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        runs.append((src / "_run.jsonl").read_text(encoding="utf-8").splitlines()[0])
    (dst / "_log.jsonl").write_text("\n".join(logs) + "\n", encoding="utf-8")
    (dst / "_run.jsonl").write_text("\n".join(runs) + "\n", encoding="utf-8")
    return dst


def outcomes(pred_dir, doc_ids):
    """원본 텍스트로 채점 → ({doc_id: 0/1 지표}, 전체 지표 행)"""
    accs, _, details = score_docs(doc_ids, {d: "all" for d in doc_ids}, cv.LABEL_DIR, pred_dir, cv.TEXT_DIR, quiet=True)
    return {d: doc_outcomes(details[d]) for d in doc_ids}, dict(accs["all"].rows())


def cmd_report(args):
    exp = cv.load_experiments(cv.EXPERIMENTS)
    fold_of = cv.read_folds(cv.CV_ROOT)
    groups = {d: r["split_group"] for d, r in cv.labeled_docs().items()}
    ids = sorted(fold_of)
    orig = {}
    for cond in CONDITIONS:
        ctype = exp["conditions"][cond]["type"]
        orig[cond] = outcomes(cv.CV_ROOT / "pooled" / cond / cv.INFER_CONDITION[ctype] / "val", ids)
    rows, L = [], ["# 보지 않은 서식 변형 강건성", "",
                   "- 학습 증강 렌더러 6종에 없는 변형으로 평가 문서 45건의 모양만 바꿔 같은 fold 모델로 다시 추론했다. "
                   "정의 · 판정 기준은 eval/cv_perturb.py 머리말(결과 전에 커밋).",
                   "- 원본 = 그룹 교차검증 출력(cv/pooled/). 채점은 둘 다 원본 텍스트 기준. 하락 = 원본 − 변형, 제조사 그룹 부트스트랩 95% 구간.",
                   ""]
    stats = PERTURB_ROOT / "make_stats.md"
    if stats.exists():
        L += ["변형 강도:", "", stats.read_text(encoding="utf-8"), ""]
    for name in SETS:
        pert, drop = {}, {}
        for cond in CONDITIONS:
            ctype = exp["conditions"][cond]["type"]
            pd = pool_perturbed(name, cond, ctype, fold_of)
            if pd is None:
                L += [f"## {name}", "", f"- {cond}: 추론이 끝나지 않음", ""]
                break
            pert[cond] = outcomes(pd, ids)
            for m, v in pert[cond][1].items():
                rows.append({"set": name, "condition": cond, "metric": m, "original": orig[cond][1].get(m), "perturbed": v})
        if len(pert) < len(CONDITIONS):
            continue
        L += [f"## {name}", "", "| 지표 | " + " | ".join(f"{c} 원본 → 변형" for c in CONDITIONS) + " |",
              "|---|" + "---|" * len(CONDITIONS)]
        for m in ("parse_rate", "schema_rate", "product_name_acc", "signal_word_acc", "ghs_f1", "hcode_f1", "cas_f1",
                  "nocas_f1", "doc_exact_ext_rate"):
            L.append(f"| {m} | " + " | ".join(f"{cv.fmt(orig[c][1].get(m), 'x')} → {cv.fmt(pert[c][1].get(m), 'x')}"
                                             for c in CONDITIONS) + " |")
        L += ["", "| 하락(원본 − 변형), doc_exact_ext | 승(원본만 맞음) · 패 · 무 | 하락 [95% 구간] |", "|---|---|---|"]
        for cond in CONDITIONS:
            r = next(x for x in compare(orig[cond][0], pert[cond][0], groups, ids) if x["metric"] == "doc_exact_ext")
            drop[cond] = {d: orig[cond][0][d]["doc_exact_ext"] - pert[cond][0][d]["doc_exact_ext"] for d in ids}
            L.append(f"| {cond} | {r['win']} · {r['loss']} · {r['tie']} | {r['mean_diff']:+.3f} "
                     f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] |")
        dd = {d: drop["r1"][d] - drop["base_fs"][d] for d in ids}
        lo, hi = bootstrap_ci(dd, groups)
        mean = sum(dd.values()) / len(dd)
        verdict = ("r1이 이 변형에 더 강건하다" if hi < 0 else
                   "r1이 이 변형에 더 약하다" if lo > 0 else "차이를 확인하지 못했다")
        L += [f"| **r1 하락 − base_fs 하락** | | **{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]** → {verdict} |", ""]
        r1_lost = [d for d in ids if drop["r1"][d] > 0]
        fs_lost = [d for d in ids if drop["base_fs"][d] > 0]
        L += [f"- 변형으로 새로 틀린 문서 — r1: {' '.join(r1_lost) or '없음'} / base_fs: {' '.join(fs_lost) or '없음'}", ""]
    cv.write_csv(PERTURB_ROOT / "scores.csv", rows, ["set", "condition", "metric", "original", "perturbed"])
    (PERTURB_ROOT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("make", help="변형 텍스트 생성 + 내용 보존 검사")
    r = sub.add_parser("run", help="fold별 추론")
    r.add_argument("--set", choices=sorted(SETS))
    sub.add_parser("report", help="채점 · 하락 폭 · 보고서")
    args = ap.parse_args(argv)
    {"make": cmd_make, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
