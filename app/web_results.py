"""
저장된 val 실험 결과 → 검토 화면(web/)용 JSON. docs/frontend_api_spec.md의 Document · CompareResult 모양을 따른다.

  python3 -m app.web_results                 # → web/public/results/documents.json · compare.json
  python3 -m app.web_results --root <폴더>   # 결과 파일을 다른 작업 폴더에서 읽을 때

- 입력: report/scores.csv, outputs/{조건}/val/({doc_id}.json · _log.jsonl · _run.jsonl), data/text/, data/splits.csv
- 모델 출력 원문을 파싱해 extraction으로 쓰고, Rule Engine(app/rules)을 돌려 핵심 5필드의 검토 상태를 붙인다.
- 파싱에 실패했거나 필수 키가 없는 출력은 목록에 넣지 않고 skipped에 사유와 함께 남긴다(조용히 버리지 않음).
- val만 다룬다. test는 저장소 밖이며 여기서 읽지 않는다.
- 원본 PDF는 커밋하지 않는다(data/raw/, 저작권). pdf_url은 "./pdfs/{doc_id}.pdf"로만 적고, 개발 · 미리보기 서버
  (web/vite.config.js)가 이 결과 파일에 있는 문서만 로컬 data/raw/에서 찾아 보여 준다. 쪽수는 로컬에 PDF가 있을 때만 센다.
나중에 app/main.py의 GET /documents · POST /compare가 같은 함수를 쓰면 된다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rules.engine import run_rules  # noqa: E402
from core.repair import repair_output  # noqa: E402
from core.schema import extract_json  # noqa: E402

CONDITIONS = ("base_zs", "base_fs", "qlora_r1", "qlora_r2", "qlora_r3", "qlora_final")
LABEL = {"base_zs": ("Base Zero-shot", "Base ZS"), "base_fs": ("Base Few-shot (k=2)", "Base FS"),
         "qlora_r1": ("QLoRA r1", "QLoRA r1"), "qlora_r2": ("QLoRA r2", "QLoRA r2"),
         "qlora_r3": ("QLoRA r3", "QLoRA r3"),
         "qlora_final": ("QLoRA 최종", "QLoRA 최종")}
# v1이 아닌 프롬프트로 만든 조건: 출력 outputs/prompt_{버전}/, 점수 report/scores_prompt_{버전}.csv(CLAUDE.md 규약).
# 비교표에는 "프롬프트 다름"으로 표시해 함께 보여 준다. r3 = r1에서 프롬프트만 v2_1(plan 6절)
OTHER_PROMPT = {"qlora_r3": "v2_1"}
DOC_ORDER = ("qlora_final", "qlora_r3", "qlora_r2", "qlora_r1", "base_fs", "base_zs")  # 문서 목록 순서: 후보 모델 먼저
CORE_FIELDS = ("product_name", "ingredients", "ghs_classification", "signal_word", "hazard_statements")
SECTION = {"product_name": "1항 가.", "ingredients": "3항", "ghs_classification": "2항 가.",
           "signal_word": "2항 나.", "hazard_statements": "2항 나."}
REQUIRED_DICTS = ("product_name", "recommended_use", "use_restrictions", "supplier", "signal_word", "list_status")
REQUIRED_LISTS = ("ingredients", "ghs_classification", "hazard_statements")

# Rule Engine 검사 → plan 8절 · 화면 상태(OK · REVIEW_REQUIRED · SOURCE_CHECK_REQUIRED + 사유 코드)
FORMAT_CHECKS = {"schema", "cas", "hcode", "empty_value"}          # 검토 필요 · FORMAT_ERROR
CONSISTENCY_CHECKS = {"ghs_signal_hcode_consistency", "phrase_to_hcode"}  # 검토 필요 · INCONSISTENT
SOURCE_CHECKS = {"not_found", "content_sum"}                       # 원문 확인 필요
# source_codes 검사: 원문에 있는 CAS · H코드가 출력에 없음 → 검토 필요 · NOT_FOUND(plan 8절 누락 재판정),
# 출력 H코드가 원문에 없음(HCODE_NOT_IN_SOURCE) → 원문 확인 필요
MISSING_CODES = {"CAS_IN_SOURCE_MISSING", "HCODE_IN_SOURCE_MISSING"}
_TOP_FIELD = re.compile(r"^\$\.([a-z_]+)")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def _fmt_time(iso: Optional[str]) -> Optional[str]:
    if not iso:
        return None
    return datetime.fromisoformat(iso).strftime("%Y.%m.%d %H:%M")


def _top_field(path: Optional[str]) -> Optional[str]:
    m = _TOP_FIELD.match(path or "")
    return m.group(1) if m else None


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def _collapse(s: str) -> str:
    """굵은 글씨가 같은 글자로 여러 번 추출된 원본 PDF("제제제제제품품품품품명")도 찾을 수 있게
    반복 글자를 하나로 줄인다. data/text(전처리 텍스트)는 이미 정리돼 있어 _squash만으로 되지만,
    pdfplumber로 직접 읽는 원본 PDF 쪽 텍스트는 이 정리를 거치지 않는다."""
    return re.sub(r"(.)\1+", r"\1", _squash(s))


def _find_page(pdf_pages: Optional[list[str]], text: Optional[str]) -> Optional[int]:
    """근거 줄(text)이 처음 나오는 원본 PDF 쪽(1부터). 줄이 여럿이면 첫 줄로 찾는다.
    쪽 텍스트가 없거나 못 찾으면 None."""
    if not pdf_pages or not text:
        return None
    key = _collapse(text.splitlines()[0])
    if not key:
        return None
    for i, page in enumerate(pdf_pages):
        if key in _collapse(page):
            return i + 1
    return None


def field_rule_results(label: dict, source_text: Optional[str], doc_id: str,
                       pdf_pages: Optional[list[str]] = None) -> dict:
    """run_rules() 결과를 핵심 5필드별 {review_status, reason_code, page, section, source_text}로 바꾼다.
    source_text는 값이 들어 있는 원문 줄 전체(app/rules/checks/evidence.py가 찾음), page는 원본 PDF에서 찾은 쪽
    (pdf_pages가 없으면 None — 전처리 텍스트만으로는 쪽 경계를 모른다)."""
    rules = run_rules(label, source_text, doc_id)
    per: dict[str, list[dict]] = {f: [] for f in CORE_FIELDS}
    for f in rules["findings"]:
        top = _top_field(f.get("field"))
        if top in per and f["severity"] in ("error", "warning"):
            per[top].append(f)
    snippets: dict[str, list[str]] = {f: [] for f in CORE_FIELDS}
    for e in rules["evidence"]:
        top = _top_field(e["field"])
        if top in snippets and e["found"] and e.get("snippet") and e["snippet"] not in snippets[top]:
            snippets[top].append(e["snippet"].strip())

    out = {}
    for field in CORE_FIELDS:
        found = per[field]
        checks = {f["check"] for f in found}
        codes = {f["code"] for f in found}
        if "source_codes" in checks and not codes & MISSING_CODES:
            checks = (checks - {"source_codes"}) | {"not_found"}
        if checks & FORMAT_CHECKS:
            status, reason = "REVIEW_REQUIRED", "FORMAT_ERROR"
        elif checks & CONSISTENCY_CHECKS:
            status, reason = "REVIEW_REQUIRED", "INCONSISTENT"
        elif codes & MISSING_CODES:
            status, reason = "REVIEW_REQUIRED", "NOT_FOUND"
        elif checks & SOURCE_CHECKS:
            status, reason = "SOURCE_CHECK_REQUIRED", None
        else:
            status, reason = "OK", None
        # signal_word · ghs_classification도 app/rules(evidence.py)가 원문 대조 대상에 넣고 있어
        # snippets[field]로 다 들어온다. 개수 상한은 성분이 많은 문서(10건 이상)에서 뒤쪽 성분의
        # CAS·함유량 근거가 잘리지 않도록 넉넉히 잡는다.
        text = "\n".join(snippets[field][:20]) or None
        out[field] = {"review_status": status, "reason_code": reason, "page": _find_page(pdf_pages, text),
                      "section": SECTION[field], "source_text": text, "messages": [f["message"] for f in found][:5]}
    return out


def _pdf_pages(pdf: Path) -> Optional[list[str]]:
    """원본 PDF 쪽별 텍스트(쪽수 · 근거 쪽 찾기용). 로컬에 없거나 읽지 못하면 None."""
    if not pdf.is_file():
        return None
    try:
        import pdfplumber
        with pdfplumber.open(pdf) as doc:
            return [page.extract_text() or "" for page in doc.pages]
    except Exception:  # noqa: BLE001 — 표시용이라 실패해도 결과 생성은 계속한다
        return None


def _missing_keys(obj: Any) -> list[str]:
    if not isinstance(obj, dict):
        return ["(JSON 객체 아님)"]
    return ([k for k in REQUIRED_DICTS if not isinstance(obj.get(k), dict)]
            + [k for k in REQUIRED_LISTS if not isinstance(obj.get(k), list)])


def build_documents(root: Path = ROOT) -> dict:
    splits = {r["doc_id"]: r for r in csv.DictReader(open(root / "data/splits.csv", encoding="utf-8-sig"))}
    sources = {}
    if (root / "data/sources.csv").exists():
        sources = {r["doc_id"]: r for r in csv.DictReader(open(root / "data/sources.csv", encoding="utf-8-sig"))}
    val_ids = sorted(d for d, r in splits.items() if r["split"] == "val")
    docs, skipped, n = [], [], 0
    pages: dict[str, Optional[list[str]]] = {}
    for cond in DOC_ORDER:
        out_dir = root / "outputs" / cond / "val"
        if not (out_dir / "_run.jsonl").exists():
            continue
        name = LABEL[cond][0]
        log = {r["doc_id"]: r for r in _read_jsonl(out_dir / "_log.jsonl")}
        for doc_id in val_ids:
            rec = log.get(doc_id) or {}
            path = out_dir / f"{doc_id}.json"
            if rec.get("skipped") or not path.exists():
                skipped.append({"doc_id": doc_id, "condition": cond, "reason": rec.get("skipped") or "출력 파일 없음"})
                continue
            obj, err = extract_json(path.read_text(encoding="utf-8"))
            missing = _missing_keys(obj) if obj is not None else []
            if obj is None or missing:
                skipped.append({"doc_id": doc_id, "condition": cond,
                                "reason": str(err) if obj is None else f"필수 키 없음: {missing}"})
                continue
            text_path = root / "data" / "text" / f"{doc_id}.txt"
            text = text_path.read_text(encoding="utf-8") if text_path.exists() else None
            obj = repair_output(obj, text)[0]  # 서빙(app/main.ingest)과 같은 출력 보정
            n += 1
            when = _fmt_time(rec.get("created_at"))
            source_file = sources.get(doc_id, {}).get("source_file")
            if source_file and doc_id not in pages:
                pages[doc_id] = _pdf_pages(root / "data" / "raw" / source_file)
            docs.append({
                "id": f"{doc_id}__{cond}", "number": n, "doc_id": doc_id, "condition": cond,
                "file_name": source_file or f"{doc_id}.pdf",
                "language": "한국어" if splits[doc_id].get("lang") == "ko" else "영어",
                "page_count": len(pages[doc_id]) if pages.get(doc_id) else None, "submission_number": doc_id, "revision_date": None,
                "extracted_at": when, "updated_at": when, "owner": "미지정",
                "split": f"val · {name}", "form": splits[doc_id].get("form"), "model_name": name,
                "generation_seconds": rec.get("gen_time_sec"), "output_tokens": rec.get("output_tokens"),
                "pdf_url": f"./pdfs/{doc_id}.pdf" if source_file else None, "source": None, "source_text": text,
                "extraction": obj, "reviews": {}, "confirmed_fields": [],
                "rule_results": field_rule_results(obj, text, doc_id, pages.get(doc_id)),
            })
    stamp = hashlib.sha256(json.dumps(docs, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    return {"generated_from": "outputs/*/val · data/text · app/rules", "split": "val", "stamp": stamp,
            "n_val_docs": len(val_ids), "documents": docs, "skipped": skipped}


def _num(v: str) -> Optional[float]:
    return float(v) if v not in ("", None) else None


def build_compare(root: Path = ROOT) -> dict:
    scores: dict[str, dict[str, str]] = {}
    for r in csv.DictReader(open(root / "report/scores.csv", encoding="utf-8-sig")):
        if r["split"] == "val" and r["subset"] == "ko":
            scores.setdefault(r["condition"], {})[r["metric"]] = r["value"]
    runs = {c: _read_jsonl(root / "outputs" / c / "val" / "_run.jsonl")[:1] for c in CONDITIONS}
    for cond, ver in OTHER_PROMPT.items():
        score_csv, run_path = root / f"report/scores_prompt_{ver}.csv", root / f"outputs/prompt_{ver}/{cond}/val/_run.jsonl"
        if cond in scores or not score_csv.exists():
            continue
        for r in csv.DictReader(open(score_csv, encoding="utf-8-sig")):
            if r["condition"] == cond and r["split"] == "val" and r["subset"] == "ko":
                scores.setdefault(cond, {})[r["metric"]] = r["value"]
        runs[cond] = _read_jsonl(run_path)[:1]
    # 최종 모델(단계 7 모델 고정) = eval/experiment.json의 어댑터 해시와 같은 어댑터로 만든 조건
    exp_path = root / "eval/experiment.json"
    final_sha = json.loads(exp_path.read_text(encoding="utf-8")).get("adapter_sha256") if exp_path.exists() else None
    exps, times = [], []
    for cond in CONDITIONS:
        s = scores.get(cond)
        if not s or not runs.get(cond):
            continue
        run = runs[cond][0]
        times.append(run.get("created_at"))
        if cond == "base_fs":
            condition = f"예시 2건(k=2): {' · '.join(run.get('fewshot_doc_ids') or [])}"
        elif cond.startswith("qlora"):
            cfg_path = root / Path(run.get("adapter") or "").parent / "config.json"
            c = json.loads(cfg_path.read_text(encoding="utf-8"))["config"] if cfg_path.exists() else {}
            condition = (f"LoRA r{c.get('lora_r', '?')} · lr {float(c.get('learning_rate', 0)):.0e} · {c.get('num_epochs', '?')} epoch".replace("e-0", "e-")
                         if c else "LoRA 어댑터")
        else:
            condition = "지시문 + 문서(예시 없음)"
        prompt = run.get("prompt_version") or "v1"
        if prompt != "v1":
            condition += f" · 프롬프트 {prompt}"
        pct = lambda k: round(_num(s.get(k)) * 100, 1) if _num(s.get(k)) is not None else None  # noqa: E731
        exps.append({
            "id": cond, "name": LABEL[cond][0], "short": LABEL[cond][1], "condition": condition,
            "parsing": pct("parse_rate"), "schema": pct("schema_rate"), "cas": pct("cas_f1"), "pair": pct("pair_f1"),
            "hcode": pct("hcode_f1"), "seconds": round(_num(s["gen_time_mean_s"]), 1),
            "tokens": round(_num(s["output_tokens_mean"])),
            # 화면 명세의 7개 값 외 추가 지표(있을 때만 표시)
            "ghs": pct("ghs_f1"), "exact": pct("doc_exact_ext_rate"), "nocas": pct("nocas_f1"),
            "input_tokens": round(_num(s["input_tokens_mean"])), "n_docs": int(_num(s["n_docs"])),
            "base_revision": run.get("base_revision"), "max_new_tokens": run.get("max_new_tokens"),
            "final": bool(final_sha) and run.get("adapter_sha256") == final_sha,
            "prompt": prompt,
        })
    other = [f"{e['name']}(프롬프트 {e['prompt']})" for e in exps if e["prompt"] != "v1"]
    note = "val(조건 선택용) 결과이며 최종 보고 수치(test)가 아닙니다."
    if other:
        note += f" {' · '.join(other)}는 다른 조건(프롬프트 v1)과 지시문이 달라 프롬프트 차이가 함께 반영된 값입니다."
    return {"split": "val", "subset": "all", "executed_at": _fmt_time(max(t for t in times if t)) if times else None,
            "note": note, "experiments": exps,
            "final_id": next((e["id"] for e in exps if e["final"]), None)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT), help="결과 파일을 읽을 저장소 폴더(기본: 이 저장소)")
    ap.add_argument("--out", default=str(ROOT / "web" / "public" / "results"))
    args = ap.parse_args(argv)
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    docs, comp = build_documents(root), build_compare(root)
    (out / "documents.json").write_text(json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "compare.json").write_text(json.dumps(comp, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"documents {len(docs['documents'])}건 · 건너뜀 {len(docs['skipped'])}건 {docs['skipped']}")
    print(f"compare 실험 {[e['id'] for e in comp['experiments']]} · 실행 {comp['executed_at']} → {out}")


if __name__ == "__main__":
    main()
