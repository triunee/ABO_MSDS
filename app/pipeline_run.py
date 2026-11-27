"""PDF 1건 → 공통 PDF 전처리 → 모델(또는 명시적 mock) → 검증 → DB → JSON.

실행: python3 -m app.pipeline_run --pdf data/raw/<파일>.pdf --model qlora
업로드 API와 일괄 평가 입력은 pipeline.extract_text.extract_pdf를 공유한다.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.db import connect, init_db, save_extraction  # noqa: E402
from app.export import export_json  # noqa: E402
from app.rules.engine import run_rules  # noqa: E402
from pipeline.extract_text import extract_pdf  # noqa: E402
from core.repair import DEFAULT_REPAIR, repair_output  # noqa: E402
from core.schema import check_schema, extract_json  # noqa: E402

def get_text(pdf_path: Path, doc_id: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """PDF에서 매번 1~3항을 만든다. doc_id는 mock 출력 선택에만 사용한다."""
    result = extract_pdf(pdf_path)
    return result["text"], result["reason"] or None


# app/model.py의 MODELS("qlora", "base")와 같은 이름 + mock. DB model_name에도 이 이름이 들어간다.
MODEL_CHOICES = ("mock", "qlora", "base")
# --saved로 저장 출력을 쓸 때 app/model.py 조건과 같은 eval 출력 폴더:
# qlora = eval/experiment.json의 어댑터(r1), base = Base few-shot(k=2).
SAVED_CONDITION = {"qlora": "qlora_r1", "base": "base_fs"}


def _mock_predict(doc_id: str) -> str:
    label_path = _REPO_ROOT / "data" / "labels" / f"{doc_id}.json"
    if not label_path.exists():
        raise FileNotFoundError(
            f"mock 모드는 data/labels/(train·val)만 씁니다. {doc_id}.json을 찾을 수 없습니다 "
            "(test 문서는 이 경로에 없습니다 — 의도된 제한, 작업지시서 2절)."
        )
    return label_path.read_text(encoding="utf-8")


def _saved_predict(model_name: str, doc_id: str) -> str:
    """eval/infer.py가 이미 만들어 둔 모델 출력(outputs/{조건}/val/{doc_id}.json)을 그대로 쓴다.
    GPU·모델 가중치 없이 실제 모델 출력으로 파이프라인 뒷단(스키마·Rule Engine·DB·JSON)을 돌릴 때 쓴다."""
    path = _REPO_ROOT / "outputs" / SAVED_CONDITION[model_name] / "val" / f"{doc_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"저장된 모델 출력이 없습니다: {path} (val 문서만 있음)")
    return path.read_text(encoding="utf-8")


def predict(text: str, model_name: str, doc_id: Optional[str], saved: bool = False) -> str:
    """모델 출력 문자열을 돌려준다.
    - mock : data/labels/{doc_id}.json을 모델 출력인 척
    - saved: 이미 돌린 실제 모델 출력(outputs/{qlora_r1|base_fs}/val/{doc_id}.json)
    - 그 외: app/model.py(양세윤)의 실시간 추론 — 어댑터·프롬프트·max_new_tokens는 eval/experiment.json 고정값"""
    if model_name == "mock":
        if not doc_id:
            raise ValueError("--model mock은 --doc-id가 있어야 합니다")
        return _mock_predict(doc_id)
    from app.model import predict as real_predict
    if saved:
        if not doc_id:
            raise ValueError("--saved는 --doc-id가 있어야 합니다")
        return _saved_predict(model_name, doc_id)
    from app.model import predict as model_predict  # torch·transformers는 실시간 추론 때만 올린다

    return model_predict(text, model_name)


def _log_failure(reason: str, pdf_path: Path, model_name: str, out_dir: Path, detail: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "_failures.csv"
    is_new = not log_path.exists()
    with log_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["timestamp", "file_name", "model_name", "reason", "detail"])
        w.writerow([datetime.now(timezone.utc).isoformat(), pdf_path.name, model_name, reason, detail])


def run_pdf(
    pdf_path: str | Path,
    model_name: str,
    doc_id: Optional[str] = None,
    db_path: str | Path = "data/db/msds.sqlite",
    out_dir: str | Path = "outputs/exports",
    saved: bool = False,
) -> tuple[Optional[int], Optional[str], Optional[str]]:
    """PDF 1건을 끝까지 처리한다. (document_id, export_path, failure_reason)을 반환한다.

    실패하면 document_id·export_path는 None이고 failure_reason에 사유가 담긴다.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)

    if not pdf_path.exists():
        _log_failure("PDF_NOT_FOUND", pdf_path, model_name, out_dir, str(pdf_path))
        return None, None, "PDF_NOT_FOUND"

    text, reason = get_text(pdf_path, doc_id)
    if text is None:
        _log_failure(reason or "TEXT_EXTRACTION_FAILED", pdf_path, model_name, out_dir)
        return None, None, reason or "TEXT_EXTRACTION_FAILED"

    try:
        raw_output = predict(text, model_name, doc_id, saved=saved)
    except (FileNotFoundError, ValueError, RuntimeError, SystemExit) as e:  # app/model.py는 모델 없음을 RuntimeError·SystemExit로 알림
        _log_failure("MODEL_UNAVAILABLE", pdf_path, model_name, out_dir, str(e))
        return None, None, f"MODEL_UNAVAILABLE: {e}"

    parsed_obj, parse_err = extract_json(raw_output)
    # 출력 보정(core/repair.py): 테이블 · Rule Engine에는 보정본, raw_json에는 모델 원본(raw_output)을 남긴다
    parsed_obj, repair_used = repair_output(parsed_obj, text)
    if parsed_obj is None:
        parsed = None
        rule_result = {"review_status": "SCHEMA_ERROR", "reason_code": "FORMAT_ERROR", "detail": parse_err}
    else:
        schema_ok, schema_errors = check_schema(parsed_obj)
        if schema_ok:
            parsed = parsed_obj
            rule_result = run_rules(parsed, source_text=text, doc_id=doc_id)
        else:
            parsed = None
            rule_result = {"review_status": "SCHEMA_ERROR", "reason_code": "SCHEMA_INVALID", "detail": schema_errors}
    rule_result["repair"] = {"version": DEFAULT_REPAIR, "rules": dict(Counter(repair_used))}

    conn = connect(db_path)
    try:
        init_db(conn)
        document_id = save_extraction(conn, pdf_path, model_name, raw_output, parsed, rule_result)
        export_path = export_json(conn, document_id, out_dir=out_dir)
    finally:
        conn.close()

    return document_id, export_path, None


def _cli() -> None:
    parser = argparse.ArgumentParser(description="PDF 1건 -> SQLite 적재 -> JSON 출력")
    parser.add_argument("--pdf", required=True, help="원본 PDF 경로 (예: data/raw/foo.pdf)")
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES,
                        help="mock=정답 라벨을 모델 출력인 척 / qlora·base=app/model.py 실시간 추론")
    parser.add_argument("--saved", action="store_true",
                        help="실시간 추론 대신 이미 돌린 모델 출력(qlora→outputs/qlora_r1, base→outputs/base_fs) 사용. GPU 불필요")
    parser.add_argument("--doc-id", default=None,
                        help="mock·--saved는 필수. 주면 data/text/{doc-id}.txt 전처리 텍스트를 우선 씀(학습·평가 입력과 동일)")
    parser.add_argument("--db", default="data/db/msds.sqlite")
    parser.add_argument("--out-dir", default="outputs/exports")
    args = parser.parse_args()

    document_id, export_path, failure_reason = run_pdf(
        args.pdf, args.model, doc_id=args.doc_id, db_path=args.db, out_dir=args.out_dir,
        saved=args.saved,
    )
    if failure_reason:
        print(f"실패: {failure_reason} ({args.out_dir}/_failures.csv에 기록)", file=sys.stderr)
        sys.exit(1)
    print(f"document_id={document_id}")
    print(f"export={export_path}")


if __name__ == "__main__":
    _cli()
