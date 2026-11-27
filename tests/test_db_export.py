"""
[김건하] app/db.py + app/export.py 테스트. 매번 임시 DB·임시 출력 폴더로 돌려 실제
DB/파일을 더럽히지 않는다. pytest tmp_path 대신 tempfile을 쓴다
("작업지시서" 2절: 설치할 패키지 없음 — pytest 미설치).

실행 (저장소 루트에서, .venv 활성화 후): python3 -m unittest discover -s tests -v

data/labels/(train·val)만 쓴다. eval/test·eval/val_en(봉인)은 어디서도 읽지 않는다.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.db import connect, init_db, list_documents, save_confirmation, save_extraction  # noqa: E402
from app.export import export_json  # noqa: E402
from src.schema import MSDSLabel  # noqa: E402

LABELS_DIR = _REPO_ROOT / "data" / "labels"
ALL_LABEL_FILES = sorted(LABELS_DIR.glob("*.json"))


def _load(doc_id: str) -> dict:
    return json.loads((LABELS_DIR / f"{doc_id}.json").read_text(encoding="utf-8"))


class DbExportTestCase(unittest.TestCase):
    """DB 테스트 공통 셋업: 임시 DB, 임시 출력 폴더, 해시용 더미 PDF."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmp.name)
        self.db_path = tmp_path / "msds.sqlite"
        self.out_dir = tmp_path / "exports"
        self.dummy_pdf = tmp_path / "dummy.pdf"
        self.dummy_pdf.write_bytes(b"%PDF-1.4 fake bytes, save_extraction only hashes this")
        self.conn = connect(self.db_path)
        init_db(self.conn)

    def tearDown(self):
        self.conn.close()
        self._tmp.cleanup()

    def _save(self, label: dict, model_name: str = "mock", pdf_path: Path | None = None) -> int:
        raw = json.dumps(label, ensure_ascii=False)
        return save_extraction(self.conn, pdf_path or self.dummy_pdf, model_name, raw, label, rule_result=None)


class TestRoundTrip(DbExportTestCase):
    def test_all_train_val_labels_round_trip(self):
        self.assertTrue(ALL_LABEL_FILES, "data/labels/*.json이 비어 있습니다")
        for path in ALL_LABEL_FILES:
            with self.subTest(doc=path.stem):
                label = json.loads(path.read_text(encoding="utf-8"))
                doc_id = self._save(label)
                out_path = export_json(self.conn, doc_id, out_dir=self.out_dir)
                exported = json.loads(Path(out_path).read_text(encoding="utf-8"))
                self.assertEqual(exported["extraction"], label, f"{path.stem} 왕복 불일치")
                MSDSLabel.model_validate(exported["extraction"])


class TestRowCounts(DbExportTestCase):
    def test_ingredient_classification_hazard_row_counts_match(self):
        label = _load("KR-KUMHO-001")
        doc_id = self._save(label)
        counts = {
            table: self.conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE document_id = ?", (doc_id,)
            ).fetchone()[0]
            for table in ("msds_ingredients", "msds_classifications", "msds_hazard_statements")
        }
        self.assertEqual(counts["msds_ingredients"], len(label["ingredients"]))
        self.assertEqual(counts["msds_classifications"], len(label["ghs_classification"]))
        self.assertEqual(counts["msds_hazard_statements"], len(label["hazard_statements"]))


class TestRevision(DbExportTestCase):
    def test_same_pdf_twice_gets_revision_1_then_2(self):
        label = _load("KR-KUMHO-001")
        id1 = self._save(label)
        id2 = self._save(label)
        rev = lambda i: self.conn.execute(  # noqa: E731
            "SELECT revision FROM msds_documents WHERE id = ?", (i,)
        ).fetchone()[0]
        self.assertEqual((rev(id1), rev(id2)), (1, 2))

        p1 = export_json(self.conn, id1, out_dir=self.out_dir)
        p2 = export_json(self.conn, id2, out_dir=self.out_dir)
        self.assertNotEqual(p1, p2)
        self.assertTrue(Path(p1).exists() and Path(p2).exists())


class TestModelSeparation(DbExportTestCase):
    def test_same_pdf_different_models_each_get_revision_1(self):
        label = _load("KR-KUMHO-001")
        id_mock = self._save(label, model_name="mock")
        id_base = self._save(label, model_name="base")
        rev = lambda i: self.conn.execute(  # noqa: E731
            "SELECT revision FROM msds_documents WHERE id = ?", (i,)
        ).fetchone()[0]
        self.assertEqual((rev(id_mock), rev(id_base)), (1, 1))


class TestBrokenModelOutput(DbExportTestCase):
    def test_unparseable_output_stores_document_row_only(self):
        doc_id = save_extraction(
            self.conn, self.dummy_pdf, "mock",
            raw_output="이것은 JSON이 아닙니다 { 깨진 출력",
            parsed=None,
            rule_result={"review_status": "SCHEMA_ERROR", "reason_code": "FORMAT_ERROR"},
        )
        for table in ("msds_ingredients", "msds_classifications", "msds_hazard_statements"):
            n = self.conn.execute(f"SELECT COUNT(*) FROM {table} WHERE document_id = ?", (doc_id,)).fetchone()[0]
            self.assertEqual(n, 0, table)

        out_path = export_json(self.conn, doc_id, out_dir=self.out_dir)
        exported = json.loads(Path(out_path).read_text(encoding="utf-8"))
        self.assertIsNone(exported["extraction"])


class TestRollback(DbExportTestCase):
    def test_error_mid_save_rolls_back_everything(self):
        label = _load("KR-KUMHO-001")
        broken = dict(label)
        broken["ingredients"] = list(label["ingredients"]) + ["dict가 아닌 항목"]  # .get()에서 AttributeError
        with self.assertRaises(AttributeError):
            self._save(broken)
        n_docs = self.conn.execute("SELECT COUNT(*) FROM msds_documents").fetchone()[0]
        self.assertEqual(n_docs, 0)


class TestOptionalHelpers(DbExportTestCase):
    def test_save_confirmation_and_list_documents(self):
        label = _load("KR-KUMHO-001")
        doc_id = self._save(label)

        review_id = save_confirmation(
            self.conn, doc_id, "product_name",
            extracted_value=label["product_name"], confirmed_value=label["product_name"],
            review_status="OK", reason_code=None,
        )
        self.assertIsInstance(review_id, int)

        docs = list_documents(self.conn)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["product_name"], label["product_name"]["value"])


class TestCliOnRealPdf(unittest.TestCase):
    """작업지시서 테스트 #7: 실제 PDF 1건을 run_pdf()(CLI가 부르는 함수)로 실행."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_run_pdf_end_to_end_with_mock_model(self):
        from app.pipeline_run import run_pdf

        with open(_REPO_ROOT / "data" / "sources.csv", encoding="utf-8-sig") as f:
            source_file = {row["doc_id"]: row["source_file"] for row in csv.DictReader(f)}["KR-KUMHO-001"]
        pdf_path = _REPO_ROOT / "data" / "raw" / source_file
        if not pdf_path.exists():
            self.skipTest(f"{pdf_path} 없음 (data/raw/는 git 제외 대상이라 로컬에 없을 수 있음)")

        document_id, export_path, failure_reason = run_pdf(
            pdf_path, "mock", doc_id="KR-KUMHO-001",
            db_path=self.tmp_path / "msds.sqlite", out_dir=self.tmp_path / "exports",
        )
        self.assertIsNone(failure_reason)
        self.assertIsInstance(document_id, int)
        exported = json.loads(Path(export_path).read_text(encoding="utf-8"))
        MSDSLabel.model_validate(exported["extraction"])

    def test_get_text_uses_real_preprocess_bypassing_text_cache(self):
        """data/text/{doc_id}.txt 캐시를 안 쓰는 경로(실시간 pdfplumber -> preprocess())를 직접 검증한다.

        위 테스트는 KR-KUMHO-001에 이미 data/text/ 캐시가 있어서 이 경로를 안 타고 통과해버린다
        (core.preprocess.preprocess()가 나중에 dict를 반환하도록 바뀌었을 때 이 구멍으로
        회귀를 놓쳤었다). doc_id=None으로 캐시를 강제로 건너뛴다.
        """
        from app.pipeline_run import get_text

        pdf_path = _REPO_ROOT / "data" / "raw" / "KR-3DSYS-001.pdf"
        if not pdf_path.exists():
            self.skipTest(f"{pdf_path} 없음 (data/raw/는 git 제외 대상이라 로컬에 없을 수 있음)")

        text, reason = get_text(pdf_path, doc_id=None)
        self.assertIsNone(reason)
        self.assertIsNotNone(text)
        self.assertIn("ColorBond", text)
        self.assertIn("제품명", text)
        # 굵은 글씨 5배 중복 복원(undouble)이 실제로 적용됐는지 — 안 됐으면 이 글자가 남아있다
        self.assertNotIn("물물물물물", text)


if __name__ == "__main__":
    unittest.main()


class TestRunPdfRepair(unittest.TestCase):
    """[양세윤] run_pdf: 테이블 · Rule Engine에는 보정본(core/repair.py), raw_json에는 모델 원본이 들어가는지.
    PDF 추출 · 모델은 가짜로 바꿔 끼워 pdfplumber · GPU 없이 돈다."""

    def test_repaired_tables_raw_json_original(self):
        import importlib
        from unittest import mock

        # pipeline.extract_text가 불러올 때 pdfplumber를 import한다. 이 테스트는 PDF를 읽지 않으므로 가짜로 둔다
        with mock.patch.dict(sys.modules, {"pdfplumber": sys.modules.get("pdfplumber") or mock.MagicMock()}):
            pipeline_run = importlib.import_module("app.pipeline_run")

        label = _load(ALL_LABEL_FILES[0].stem)
        label["hazard_statements"] = [{"code": "17", "text": "알레르기성 피부 반응을 일으킬 수 있음"}]
        label["list_status"]["hazard_statements"] = "기재"
        raw = json.dumps(label, ensure_ascii=False)
        with tempfile.TemporaryDirectory() as tmp:
            pdf = Path(tmp) / "x.pdf"
            pdf.write_bytes(b"%PDF fake")
            with mock.patch.object(pipeline_run, "get_text", return_value=("원문 텍스트", None)), \
                    mock.patch.object(pipeline_run, "predict", return_value=raw):
                doc_id, _, fail = pipeline_run.run_pdf(pdf, "qlora", db_path=Path(tmp) / "t.db", out_dir=Path(tmp) / "out")
            self.assertIsNone(fail)
            conn = connect(Path(tmp) / "t.db")
            row = conn.execute("SELECT raw_json, rule_json FROM msds_documents WHERE id = ?", (doc_id,)).fetchone()
            codes = [r["code"] for r in conn.execute("SELECT code FROM msds_hazard_statements WHERE document_id = ?",
                                                    (doc_id,))]
            conn.close()
        self.assertEqual(row["raw_json"], raw)                       # 모델 원본 그대로
        self.assertEqual(codes, [None])                               # 깨진 코드 "17" → null(R3)
        self.assertEqual(json.loads(row["rule_json"])["repair"], {"version": "v1", "rules": {"R3": 1}})
