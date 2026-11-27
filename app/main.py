"""
[양세윤] FastAPI — 검토 화면(web/)의 API 모드(workspace.html?mode=api)가 부르는 서버. SQLite(app/db.py)에 적재 · 확정 저장.

  python3 -m app.main load-val     # 저장된 val 출력(qlora_r1 · base_fs)을 DB에 적재 — "재생 결과", 이미 있으면 건너뜀
  python3 -m app.main serve        # http://127.0.0.1:8000
  python3 -m app.main serve --preload   # 시연용: 켤 때 모델을 미리 올려 첫 업로드가 모델 적재를 기다리지 않게
  cd web && npm run dev            # → http://localhost:5173/workspace.html?mode=api (vite가 API 경로를 8000으로 넘김)

엔드포인트(plan 7절)
  GET  /documents       DB 문서 → 화면 Document 모양(extraction · rule_results · reviews · confirmed_fields)
  POST /extract         PDF 업로드 → 1~3항 텍스트 → 모델(r1 QLoRA 또는 Base few-shot, app/model.py) → Rule Engine → 4개 테이블 적재
  POST /confirm         담당자 확정값 → msds_reviews
  POST /owner           문서 담당자 지정(팀원 목록 TEAM 또는 미지정) → meta/{document_id}.json의 owner
  POST /compare         val 실험 비교표(app/web_results.build_compare — test 수치 아님)
  POST /ask             MSDS 1~16항 RAG 질의응답(근거 · LangGraph 실행 경로 포함)
  GET  /files/{id}.pdf  업로드한 PDF(원문 보기)

저장 위치는 data/db/(git 제외): msds.sqlite, uploads/(업로드 PDF), meta/{document_id}.json(1~3항 텍스트 · 근거 ·
쪽수 · 출처 · 생성 시간). 테이블 · 열은 동결이라, 화면에 필요한데 테이블에 없는 값은 meta에 둔다.

msds_reviews의 field_path: 필드 키(product_name · ingredients · …) 한 행 + 필요할 때 list_status.{키} ·
{키}.evidence · {키}.resolved 행. 한 번 저장한 행은 같은 confirmed_at으로 묶고, 화면에는 마지막 저장분을 보여 준다.

- 재생 결과(load-val)는 모델을 다시 돌린 것이 아니라 저장된 val 출력을 적재한 것이다. 화면의 구분 · 모델 칸에 표시한다.
- test 문서는 이 서버에 올리지 않는다. report/test_manifest.csv(봉인 해시)에 있는 파일은 업로드를 거부한다.
- 모델 출력이 JSON이 아니어도 DB에는 원본 출력으로 적재하고(조용히 버리지 않음) 요청에는 실패로 답한다.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

from app.db import connect, init_db, save_confirmation, save_extraction  # noqa: E402
from app.pipeline_run import _log_failure, get_text  # noqa: E402
from app.rules.engine import run_rules  # noqa: E402
from app.web_results import CORE_FIELDS, _missing_keys, _pdf_pages, _read_jsonl, build_compare, field_rule_results  # noqa: E402
from core.repair import DEFAULT_REPAIR, repair_output  # noqa: E402
from core.schema import check_schema, extract_json  # noqa: E402

DB_DIR = ROOT / "data" / "db"
DB_PATH = DB_DIR / "msds.sqlite"
UPLOADS = DB_DIR / "uploads"
META = DB_DIR / "meta"
TEST_MANIFEST = ROOT / "report" / "test_manifest.csv"
MAX_UPLOAD = 30 * 1024 * 1024
MODEL_LABEL = {"qlora": "QLoRA r1", "base": "Base few-shot (k=2)"}
REPLAY = {"qlora_r1": "qlora", "base_fs": "base"}  # 저장된 val 출력 폴더 → DB model_name
_SHA = re.compile(r"\b[0-9a-f]{64}\b")

app = FastAPI(title="MSDS 1~3항 추출 · 검토 API")


# ---------------------------------------------------------------- 저장소
def db():
    conn = connect(DB_PATH)
    init_db(conn)
    return conn


TEAM = ("어채은", "양세윤", "강덕우", "김건하")  # 문서 담당자로 고를 수 있는 사람(plan 11절 역할표)
NO_OWNER = "미지정"


def write_meta(document_id: int, meta: dict) -> None:
    META.mkdir(parents=True, exist_ok=True)
    (META / f"{document_id}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def read_meta(document_id: int) -> dict:
    path = META / f"{document_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def sealed_hashes() -> set[str]:
    """봉인된 test 파일 해시(report/test_manifest.csv 안의 sha256 전부). 파일이 없으면 빈 집합."""
    return set(_SHA.findall(TEST_MANIFEST.read_text(encoding="utf-8"))) if TEST_MANIFEST.exists() else set()


def ingest(conn, pdf_path: Path, model_name: str, raw: str, text: str, key: str, meta: dict) -> tuple[int, bool]:
    """모델 출력 1건 → 4개 테이블 + meta. → (document_id, 화면에 보일 수 있는지).
    DB 적재 규칙은 app/pipeline_run.run_pdf와 같다(스키마를 어기면 문서 행만, rule_json에 사유).
    출력 보정(core/repair.py)은 run_pdf와 같다: 테이블 · Rule Engine · 화면에는 보정본, raw_json에는 모델 원본."""
    obj, err = extract_json(raw)
    obj, repair_used = repair_output(obj, text)
    if obj is None:
        parsed, rule_result = None, {"review_status": "SCHEMA_ERROR", "reason_code": "FORMAT_ERROR", "detail": str(err)}
    else:
        ok, errors = check_schema(obj)
        parsed = obj if ok else None
        rule_result = (run_rules(obj, source_text=text, doc_id=key) if ok
                       else {"review_status": "SCHEMA_ERROR", "reason_code": "SCHEMA_INVALID", "detail": errors})
    rule_result["repair"] = {"version": DEFAULT_REPAIR, "rules": {r: repair_used.count(r) for r in sorted(set(repair_used))}}
    shown = obj is not None and not _missing_keys(obj)
    pages = _pdf_pages(pdf_path)
    document_id = save_extraction(conn, pdf_path, model_name, raw, parsed, rule_result)
    write_meta(document_id, {**meta, "key": key, "source_text": text, "page_count": len(pages) if pages else None,
                             "rule_results": field_rule_results(obj, text, key, pages) if shown else {}})
    return document_id, shown


# ---------------------------------------------------------------- DB → 화면 Document
def _load(value: Optional[str]) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except ValueError:
        return value  # "기재" 같은 문자열은 그대로 저장돼 있다


def _fmt(iso: Optional[str]) -> Optional[str]:
    return datetime.fromisoformat(iso).astimezone().strftime("%Y.%m.%d %H:%M") if iso else None


def latest_reviews(conn, document_id: int) -> tuple[dict, list, Optional[str]]:
    """마지막 저장분(같은 confirmed_at) → (reviews, confirmed_fields, 저장 시각)."""
    rows = conn.execute("SELECT * FROM msds_reviews WHERE document_id = ? ORDER BY id", (document_id,)).fetchall()
    if not rows:
        return {}, [], None
    last = max(r["confirmed_at"] for r in rows)
    batch = {r["field_path"]: r for r in rows if r["confirmed_at"] == last}
    reviews, confirmed = {}, []
    for key in CORE_FIELDS:
        if key not in batch:
            continue
        review = {"confirmed_value": _load(batch[key]["confirmed_value"]), "confirmed_at": last}
        if f"list_status.{key}" in batch:
            review["list_status"] = batch[f"list_status.{key}"]["confirmed_value"]
        if f"{key}.evidence" in batch:
            review["evidence"] = _load(batch[f"{key}.evidence"]["confirmed_value"])
        if f"{key}.resolved" in batch:
            review["resolved"] = _load(batch[f"{key}.resolved"]["confirmed_value"]) is True
        reviews[key] = review
        confirmed.append(key)
    return reviews, confirmed, last


def to_document(conn, row) -> tuple[Optional[dict], Optional[str]]:
    obj, err = extract_json(row["raw_json"])
    if obj is None or _missing_keys(obj):
        return None, str(err) if obj is None else f"필수 키 없음: {_missing_keys(obj)}"
    meta = read_meta(row["id"])
    reviews, confirmed, saved_at = latest_reviews(conn, row["id"])
    replay = meta.get("origin") == "val_replay"
    label = MODEL_LABEL.get(row["model_name"], row["model_name"])
    key = meta.get("key")
    return {
        "id": str(row["id"]), "number": row["id"], "doc_id": key, "file_name": row["file_name"],
        "language": meta.get("language") or "미확인", "page_count": meta.get("page_count"),
        "submission_number": key or "—", "revision_date": None, "revision": row["revision"],
        "extracted_at": _fmt(meta.get("generated_at") or row["extracted_at"]),
        "updated_at": _fmt(saved_at or row["extracted_at"]), "owner": meta.get("owner") or NO_OWNER,
        "split": "val · 재생 결과" if replay else "업로드", "form": meta.get("form"),
        "model_name": f"{label} (저장된 val 출력)" if replay else label, "replay": replay,
        "generation_seconds": meta.get("seconds"), "output_tokens": meta.get("output_tokens"),
        "pdf_url": f"./pdfs/{key}.pdf" if replay else f"/files/{row['id']}.pdf",
        "source": None, "source_text": meta.get("source_text"), "extraction": obj,
        "reviews": reviews, "confirmed_fields": confirmed, "rule_results": meta.get("rule_results") or {},
    }, None


def list_all(conn) -> tuple[list[dict], list[dict]]:
    docs, skipped = [], []
    for row in conn.execute("SELECT * FROM msds_documents ORDER BY id DESC").fetchall():
        doc, reason = to_document(conn, row)
        if doc:
            docs.append(doc)
        else:
            skipped.append({"document_id": row["id"], "file_name": row["file_name"], "model_name": row["model_name"],
                            "reason": reason})
    return docs, skipped


# ---------------------------------------------------------------- 엔드포인트
@app.get("/documents")
def documents():
    conn = db()
    try:
        docs, skipped = list_all(conn)
    finally:
        conn.close()
    return {"documents": docs, "skipped": skipped}


DATA_TABLES = ("msds_documents", "msds_ingredients", "msds_classifications",
               "msds_hazard_statements", "msds_reviews")


@app.get("/data")
def database_rows(table: str = "msds_documents", limit: int = Query(25, ge=1, le=100),
                  offset: int = Query(0, ge=0)):
    """SQLite 적재 상태를 읽기 전용으로 표시한다. 봉인된 test 파일 해시는 제외한다."""
    if table not in DATA_TABLES:
        raise HTTPException(422, f"조회할 수 있는 테이블: {', '.join(DATA_TABLES)}")
    blocked = sorted(sealed_hashes())
    where = f" WHERE d.file_hash NOT IN ({','.join('?' for _ in blocked)})" if blocked else ""
    tables = []
    conn = db()
    try:
        for name in DATA_TABLES:
            joined = ("msds_documents AS d" if name == "msds_documents"
                      else f"{name} AS t JOIN msds_documents AS d ON t.document_id = d.id")
            count = conn.execute(f"SELECT COUNT(*) FROM {joined}{where}", blocked).fetchone()[0]
            tables.append({"name": name, "count": count})
        columns = [{"name": row["name"], "type": row["type"]}
                   for row in conn.execute(f"PRAGMA table_info({table})")]
        joined = ("msds_documents AS d" if table == "msds_documents"
                  else f"{table} AS t JOIN msds_documents AS d ON t.document_id = d.id")
        alias = "d" if table == "msds_documents" else "t"
        rows = [dict(row) for row in conn.execute(
            f"SELECT {alias}.* FROM {joined}{where} ORDER BY {alias}.id DESC LIMIT ? OFFSET ?",
            [*blocked, limit, offset])]
    finally:
        conn.close()
    return {"table": table, "tables": tables, "columns": columns, "rows": rows,
            "total": next(item["count"] for item in tables if item["name"] == table),
            "limit": limit, "offset": offset}


@app.post("/extract")
def extract(file: UploadFile = File(...), model: str = Form("qlora")):
    from app.model import MODELS, predict_detail  # torch · transformers는 추출할 때만 올린다

    if model not in MODELS:
        raise HTTPException(422, f"model은 {MODELS} 중 하나여야 합니다.")
    data = file.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD or not data.startswith(b"%PDF-"):
        raise HTTPException(422, "30MB 이하의 유효한 PDF만 올릴 수 있습니다.")
    sha = hashlib.sha256(data).hexdigest()
    if sha in sealed_hashes():
        raise HTTPException(403, "봉인된 test 문서입니다. 이 화면에서는 test 문서를 다루지 않습니다.")
    name = Path(file.filename or "upload.pdf").name
    pdf_path = UPLOADS / sha[:16] / name
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    if not pdf_path.exists():
        pdf_path.write_bytes(data)

    text, reason = get_text(pdf_path, None)
    if text is None:
        _log_failure(reason or "TEXT_EXTRACTION_FAILED", pdf_path, model, DB_DIR)
        raise HTTPException(422, f"1~3항 텍스트를 뽑지 못했습니다: {reason} (data/db/_failures.csv에 기록)")
    try:
        out = predict_detail(text, model)
    except (SystemExit, RuntimeError) as issue:  # 실험 고정 · 어댑터 · 스냅샷 확인 실패
        raise HTTPException(503, f"모델을 올리지 못했습니다: {issue}") from None

    conn = db()
    try:
        document_id, shown = ingest(conn, pdf_path, model, out["raw"], text, Path(name).stem, {
            "origin": "upload", "seconds": out["seconds"], "input_tokens": out["input_tokens"],
            "output_tokens": out["output_tokens"], "hit_max_new_tokens": out["hit_max_new_tokens"],
            "prompt_version": out["prompt_version"], "base": out["base"], "adapter": out["adapter"],
            "fewshot_doc_ids": out["fewshot_doc_ids"], "generated_at": datetime.now(timezone.utc).isoformat()})
        if not shown:
            raise HTTPException(422, f"모델 출력이 올바른 JSON이 아닙니다. DB에는 원본 출력으로 적재했습니다(document_id={document_id}).")
        row = conn.execute("SELECT * FROM msds_documents WHERE id = ?", (document_id,)).fetchone()
        doc, _ = to_document(conn, row)
    finally:
        conn.close()
    return {"document": doc}


@app.post("/confirm")
def confirm(body: dict = Body(...)):
    try:
        document_id = int(body.get("document_id"))
    except (TypeError, ValueError):
        raise HTTPException(422, "document_id가 필요합니다.") from None
    reviews = body.get("reviews") or {}
    keys = [k for k in body.get("confirmed_fields") or [] if k in CORE_FIELDS and isinstance(reviews.get(k), dict)]
    if not keys:
        raise HTTPException(422, "확정한 필드가 없습니다.")
    conn = db()
    try:
        row = conn.execute("SELECT raw_json FROM msds_documents WHERE id = ?", (document_id,)).fetchone()
        if row is None:
            raise HTTPException(404, f"문서가 없습니다: {document_id}")
        extracted, _ = extract_json(row["raw_json"])
        extracted = extracted or {}
        rules = read_meta(document_id).get("rule_results") or {}
        at = datetime.now(timezone.utc).isoformat()
        n = 0
        for key in keys:
            review, rule = reviews[key], rules.get(key) or {}
            status, reason = rule.get("review_status"), rule.get("reason_code")
            items = [(key, extracted.get(key), review.get("confirmed_value"))]
            if "list_status" in review:
                items.append((f"list_status.{key}", (extracted.get("list_status") or {}).get(key), review["list_status"]))
            if review.get("evidence"):
                items.append((f"{key}.evidence", None, review["evidence"]))
            if "resolved" in review:
                items.append((f"{key}.resolved", None, bool(review["resolved"])))
            for path, before, after in items:
                save_confirmation(conn, document_id, path, before, after, status, reason, confirmed_at=at)
                n += 1
    finally:
        conn.close()
    return {"document_id": document_id, "saved_rows": n, "confirmed_fields": keys, "confirmed_at": at}


@app.post("/owner")
def set_owner(body: dict = Body(...)):
    try:
        document_id = int(body.get("document_id"))
    except (TypeError, ValueError):
        raise HTTPException(422, "document_id가 필요합니다.") from None
    owner = body.get("owner") or NO_OWNER
    if owner != NO_OWNER and owner not in TEAM:
        raise HTTPException(422, f"담당자는 {' · '.join(TEAM)} 또는 {NO_OWNER}만 지정할 수 있습니다.")
    conn = db()
    try:
        if conn.execute("SELECT 1 FROM msds_documents WHERE id = ?", (document_id,)).fetchone() is None:
            raise HTTPException(404, f"문서가 없습니다: {document_id}")
    finally:
        conn.close()
    meta = read_meta(document_id)
    if owner == NO_OWNER:
        meta.pop("owner", None)
    else:
        meta["owner"] = owner
    write_meta(document_id, meta)
    return {"document_id": document_id, "owner": owner}


@app.post("/compare")
def compare(body: dict = Body(default={})):
    result = build_compare(ROOT)
    if (body or {}).get("subset", "all") != "all":
        result["note"] += " 문서 유형별 비교는 아직 없어 전체 기준으로 보여 줍니다."
    return result


@app.post("/ask")
def ask_msds(body: dict = Body(...)):
    """RAG 인덱스에 등록된 MSDS만 근거로 답한다. 봉인된 test는 인덱싱하지 않는다."""
    question = body.get("question") if isinstance(body, dict) else None
    if not isinstance(question, str) or not question.strip():
        raise HTTPException(422, "질문을 입력해 주세요.")
    question = question.strip()
    if len(question) > 1000:
        raise HTTPException(422, "질문은 1,000자 이하로 입력해 주세요.")
    from app.rag.service import RagUnavailable, get_service

    try:
        return get_service().ask(question)
    except RagUnavailable as issue:
        raise HTTPException(503, str(issue)) from None


@app.get("/files/{document_id}.pdf")
def pdf_file(document_id: int):
    conn = db()
    try:
        row = conn.execute("SELECT file_name, file_hash FROM msds_documents WHERE id = ?", (document_id,)).fetchone()
    finally:
        conn.close()
    path = UPLOADS / row["file_hash"][:16] / row["file_name"] if row else None
    if path is None or not path.is_file():
        raise HTTPException(404, "업로드한 PDF가 이 서버에 없습니다.")
    return FileResponse(path, media_type="application/pdf", headers={"Content-Disposition": "inline"})


# ---------------------------------------------------------------- 저장된 val 출력 적재(재생 결과)
def load_val() -> None:
    splits = {r["doc_id"]: r for r in csv.DictReader(open(ROOT / "data/splits.csv", encoding="utf-8-sig"))}
    sources = {r["doc_id"]: r for r in csv.DictReader(open(ROOT / "data/sources.csv", encoding="utf-8-sig"))}
    val_ids = sorted(d for d, r in splits.items() if r["split"] == "val")
    done = set()
    if META.exists():
        for p in META.glob("*.json"):
            m = json.loads(p.read_text(encoding="utf-8"))
            if m.get("origin") == "val_replay":
                done.add((m.get("key"), m.get("condition")))
    conn = db()
    added, skipped = 0, []
    try:
        for cond, model_name in REPLAY.items():
            out_dir = ROOT / "outputs" / cond / "val"
            log = {r["doc_id"]: r for r in _read_jsonl(out_dir / "_log.jsonl")}
            for doc_id in val_ids:
                if (doc_id, cond) in done:
                    continue
                out, text_path = out_dir / f"{doc_id}.json", ROOT / "data" / "text" / f"{doc_id}.txt"
                pdf = ROOT / "data" / "raw" / sources.get(doc_id, {}).get("source_file", f"{doc_id}.pdf")
                missing = [p.relative_to(ROOT).as_posix() for p in (out, text_path, pdf) if not p.is_file()]
                if missing:
                    skipped.append((cond, doc_id, f"파일 없음: {missing}"))
                    continue
                rec = log.get(doc_id) or {}
                _, shown = ingest(conn, pdf, model_name, out.read_text(encoding="utf-8"),
                                  text_path.read_text(encoding="utf-8"), doc_id, {
                                      "origin": "val_replay", "condition": cond, "seconds": rec.get("gen_time_sec"),
                                      "input_tokens": rec.get("input_tokens"), "output_tokens": rec.get("output_tokens"),
                                      "generated_at": (datetime.fromisoformat(rec["created_at"]).astimezone().isoformat()
                                                       if rec.get("created_at") else None),
                                      "language": "한국어" if splits[doc_id].get("lang") == "ko" else "영어",
                                      "form": splits[doc_id].get("form")})
                added += 1
                if not shown:
                    skipped.append((cond, doc_id, "DB에는 적재, 화면에는 안 보임(모델 출력 JSON 파싱 실패)"))
    finally:
        conn.close()
    print(f"적재 {added}건 → {DB_PATH} (이미 있던 {len(done)}건은 건너뜀)")
    for s in skipped:
        print("  [참고]", *s)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["serve", "load-val"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--preload", action="store_true",
                    help="serve: 켤 때 모델(베이스 + r1 어댑터)을 미리 올린다. 첫 업로드 대기를 없애는 시연용")
    args = ap.parse_args(argv)
    if args.command == "load-val":
        load_val()
    else:
        import uvicorn

        if args.preload:
            from app.model import warm_up

            print("모델을 미리 올리는 중(베이스 4bit + r1 어댑터)…", flush=True)
            print(f"모델 준비 완료 {warm_up()}초 — 첫 업로드부터 생성 시간만 걸린다", flush=True)
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
