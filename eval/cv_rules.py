"""
교차검증 출력에 Rule Engine(app/rules/)을 돌려, 모델이 틀린 문서 · 필드를 검토 상태가 얼마나 잡아내는지 잰다.

  python3 eval/cv.py report        # 먼저: 모아서 채점(cv/pooled/ · 채점 상세)
  python3 eval/cv_rules.py         # → cv/rules.csv · cv/rules.md

- 틀림: 채점 상세의 문서 완전 정답(확장)이 0인 문서. 필드는 eval/paired.py의 필드별 맞음 · 틀림.
- 표시: review_status가 OK가 아닌 문서. 필드는 그 필드 경로에 error · warning finding이 있는 경우(info는 표시로 치지 않음).
- 정답 라벨을 그대로 넣은 행(gold)은 원문 모순 · 검사 잡음으로 생기는 표시의 바닥값이다.
- 파싱 실패 출력은 빈 dict로 넣는다(스키마 오류로 표시된다). 실패 건을 분모에서 빼지 않는다.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.rules.engine import run_rules  # noqa: E402
from core.schema import extract_json  # noqa: E402
from eval import cv  # noqa: E402
from eval.paired import doc_outcomes, read_detail  # noqa: E402
from eval.score import DETAIL_FILE  # noqa: E402

# 채점 필드 → Rule Engine finding 경로의 최상위 키. 성분은 cas · pair · nocas 중 하나라도 틀리면 틀림
FIELD_MAP = {"product_name": ["product_name_acc"], "ingredients": ["cas", "pair", "nocas"],
             "ghs_classification": ["ghs_classification"], "signal_word": ["signal_word_acc"],
             "hazard_statements": ["hazard_statements"]}
_TOP = re.compile(r"^\$\.?([A-Za-z_]*)\.?([A-Za-z_]*)")


def top_field(path):
    """'$.ghs_classification.2.category' → ghs_classification, '$.list_status.ingredients' → ingredients, '$' → ''"""
    m = _TOP.match(path or "$")
    if not m:
        return ""
    return m.group(2) if m.group(1) == "list_status" else m.group(1)


def judge(pred, doc_id):
    r = run_rules(pred, (cv.TEXT_DIR / f"{doc_id}.txt").read_text(encoding="utf-8"), doc_id)
    fields = {top_field(f["field"]) for f in r["findings"] if f["severity"] != "info"}
    return r["review_status"], fields


def tally(rows):
    """문서별 (틀림, 표시, 필드 틀림 집합, 필드 표시 집합) → 지표 행"""
    n = len(rows)
    err = [r for r in rows if r["wrong"]]
    flag = [r for r in rows if r["status"] != "OK"]
    tp = [r for r in err if r["status"] != "OK"]
    ok = [r for r in rows if r["status"] == "OK"]
    out = {"n_docs": n, "n_wrong": len(err), "n_flagged": len(flag), "n_caught": len(tp),
           "recall": len(tp) / len(err) if err else None,
           "precision": len(tp) / len(flag) if flag else None,
           "pass_rate": len(ok) / n,
           "ok_acc": sum(not r["wrong"] for r in ok) / len(ok) if ok else None,
           "missed": " ".join(r["doc_id"] for r in err if r["status"] == "OK")}
    for f in FIELD_MAP:
        fe = [r for r in rows if f in r["wrong_fields"]]
        fr = [r for r in rows if f not in r["wrong_fields"]]
        out[f"{f}_wrong"] = len(fe)
        out[f"{f}_caught"] = sum(f in r["flag_fields"] for r in fe)
        out[f"{f}_false_alarm"] = sum(f in r["flag_fields"] for r in fr)
    return out


def condition_rows(pred_dir, doc_ids):
    _, details = read_detail(pred_dir / DETAIL_FILE)
    rows = []
    for d in doc_ids:
        o = doc_outcomes(details[d])
        pred = extract_json((pred_dir / f"{d}.json").read_text(encoding="utf-8"))[0] or {}
        status, fields = judge(pred, d)
        rows.append({"doc_id": d, "wrong": not o["doc_exact_ext"], "status": status, "flag_fields": fields,
                     "wrong_fields": {f for f, ms in FIELD_MAP.items() if any(not o[m] for m in ms)}})
    return rows


def gold_rows(doc_ids):
    rows = []
    for d in doc_ids:
        status, fields = judge(json.loads((cv.LABEL_DIR / f"{d}.json").read_text(encoding="utf-8")), d)
        rows.append({"doc_id": d, "wrong": False, "status": status, "flag_fields": fields, "wrong_fields": set()})
    return rows


def pct(v):
    return "—" if v is None else f"{v:.1%}"


def write_md(path, results):
    L = ["# Rule Engine 오류 탐지 (교차검증 출력)", "",
         "- 대상: 그룹 5-fold 교차검증 출력(cv/pooled/) 45건 × 조건별. Rule Engine은 app/rules/(검사 9개) 그대로.",
         "- 틀림 = 문서 완전 정답(확장) 0. 표시 = review_status가 OK가 아님(NEEDS_REVIEW · INCONSISTENT · SCHEMA_ERROR).",
         "- gold 행은 정답 라벨을 그대로 넣은 결과다. 원문 모순 · 검사 잡음으로 생기는 표시의 바닥값이다.",
         "- 생성: `python3 eval/cv_rules.py`", "",
         "## 문서 단위", "",
         "| 조건 | 틀린 문서 | 표시 문서 | 탐지율(틀린 것 중 표시) | 정밀도(표시 중 틀림) | OK 통과율 | OK 문서 정확도 |",
         "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        L.append(f"| {name} | {r['n_wrong']}/{r['n_docs']} | {r['n_flagged']} | {pct(r['recall'])} ({r['n_caught']}/{r['n_wrong']}) "
                 f"| {pct(r['precision'])} | {pct(r['pass_rate'])} | {pct(r['ok_acc'])} |")
    L += ["", "## 필드 단위 (틀린 필드 중 그 필드에 표시가 붙은 수 / 맞은 필드에 붙은 표시 수)", "",
          "| 조건 | " + " | ".join(FIELD_MAP) + " |", "|---|" + "---|" * len(FIELD_MAP)]
    for name, r in results.items():
        L.append(f"| {name} | " + " | ".join(f"{r[f + '_caught']}/{r[f + '_wrong']} · 오경보 {r[f + '_false_alarm']}"
                                             for f in FIELD_MAP) + " |")
    L += ["", "## OK로 통과한 틀린 문서", ""]
    for name, r in results.items():
        if name != "gold":
            L.append(f"- {name}: {r['missed'] or '없음'}")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "cv"))
    ap.add_argument("--experiments", default=str(cv.EXPERIMENTS))
    args = ap.parse_args(argv)
    root = Path(args.root)
    exp = cv.load_experiments(args.experiments)
    fold_of = cv.read_folds(root)
    doc_ids = sorted(fold_of)
    results = {"gold": tally(gold_rows(doc_ids))}
    for name, c in exp["conditions"].items():
        pred_dir = root / "pooled" / name / cv.INFER_CONDITION[c["type"]] / "val"  # cv.py report가 모아 둔 폴더(읽기만)
        if not (pred_dir / DETAIL_FILE).exists():
            print(f"[건너뜀] {name}: 채점 상세 없음 — python3 eval/cv.py report 먼저")
            continue
        results[name] = tally(condition_rows(pred_dir, doc_ids))
    cv.write_csv(root / "rules.csv", [{"condition": n, **r} for n, r in results.items()],
                 ["condition"] + list(next(iter(results.values()))))
    write_md(root / "rules.md", results)
    print((root / "rules.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
