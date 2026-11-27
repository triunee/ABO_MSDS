"""
[김건하] Rule Engine 통합 테스트. fixtures.py의 가짜 라벨/가짜 원문만 쓴다.
실행: `cd` 저장소 루트 → `python -m unittest discover -s app/rules/tests -v`
"""
from __future__ import annotations

import unittest

from app.rules.engine import run_rules
from app.rules.tests.fixtures import BASE_SOURCE_TEXT, clean_label


def _codes(result: dict) -> set[str]:
    return {f["code"] for f in result["findings"]}


class TestOutputShape(unittest.TestCase):
    def test_top_level_keys_match_spec(self):
        result = run_rules(clean_label(), BASE_SOURCE_TEXT, doc_id="FIXTURE-001")
        expected_keys = {
            "doc_id", "review_status", "schema_valid", "source_text_available",
            "checks_run", "findings", "evidence", "summary",
        }
        self.assertEqual(set(result.keys()), expected_keys)
        self.assertEqual(result["doc_id"], "FIXTURE-001")
        self.assertEqual(len(result["checks_run"]), 10)
        self.assertEqual(set(result["summary"].keys()), {"n_errors", "n_warnings", "n_info"})

    def test_finding_and_evidence_shape(self):
        label = clean_label()
        label["ingredients"][0]["cas_number"] = "64-17-6"  # 체크디지트 오류 유발
        result = run_rules(label, BASE_SOURCE_TEXT)
        self.assertTrue(result["findings"])
        finding = result["findings"][0]
        self.assertEqual(
            set(finding.keys()), {"check", "severity", "code", "field", "message", "detail"}
        )
        self.assertTrue(result["evidence"])
        ev = result["evidence"][0]
        self.assertEqual(set(ev.keys()), {"field", "value", "found", "match_type", "snippet"})


class TestCleanBaseline(unittest.TestCase):
    def test_ok_status_no_findings(self):
        result = run_rules(clean_label(), BASE_SOURCE_TEXT, doc_id="FIXTURE-OK")
        self.assertEqual(result["review_status"], "OK")
        self.assertEqual(result["findings"], [])
        self.assertTrue(result["schema_valid"])
        self.assertTrue(result["source_text_available"])


class TestNoSourceText(unittest.TestCase):
    def test_evidence_and_not_found_skip_gracefully(self):
        result = run_rules(clean_label(), None, doc_id="FIXTURE-NOSRC")
        self.assertFalse(result["source_text_available"])
        self.assertEqual(result["evidence"], [])
        self.assertIn("EVIDENCE_SKIPPED_NO_SOURCE", _codes(result))
        # 정보성 1건만 있으므로 review_status는 여전히 OK
        self.assertEqual(result["review_status"], "OK")
        self.assertEqual(result["summary"], {"n_errors": 0, "n_warnings": 0, "n_info": 1})


class TestSchemaCheck(unittest.TestCase):
    def test_missing_required_field_is_schema_error(self):
        label = clean_label()
        del label["list_status"]
        result = run_rules(label, BASE_SOURCE_TEXT)
        self.assertEqual(result["review_status"], "SCHEMA_ERROR")
        self.assertFalse(result["schema_valid"])
        self.assertIn("SCHEMA_FIELD_INVALID", _codes(result))


class TestEmptyValueCheck(unittest.TestCase):
    def test_flagged_when_value_is_sentinel_but_status_is_filled(self):
        label = clean_label()
        label["use_restrictions"] = {"value": "자료없음", "source_status": "기재"}
        result = run_rules(label, BASE_SOURCE_TEXT)
        self.assertIn("EMPTY_VALUE_SUSPECT", _codes(result))
        self.assertEqual(result["review_status"], "NEEDS_REVIEW")


class TestNotFoundCheck(unittest.TestCase):
    def test_flagged_when_value_absent_from_source(self):
        label = clean_label()
        label["product_name"]["value"] = "원문에없는이름"
        result = run_rules(label, BASE_SOURCE_TEXT)
        self.assertIn("VALUE_NOT_FOUND_IN_SOURCE", _codes(result))
        self.assertEqual(result["review_status"], "NEEDS_REVIEW")


class TestSourceCodesCheck(unittest.TestCase):
    def test_clean_label_has_no_source_code_findings(self):
        result = run_rules(clean_label(), BASE_SOURCE_TEXT)
        self.assertFalse({c for c in _codes(result) if "IN_SOURCE" in c})

    def test_missing_ingredient_cas_flagged(self):
        label = clean_label()
        del label["ingredients"][1]  # 67-56-1은 원문에 있는데 출력에서 빠짐
        result = run_rules(label, BASE_SOURCE_TEXT)
        f = [x for x in result["findings"] if x["code"] == "CAS_IN_SOURCE_MISSING"]
        self.assertEqual(len(f), 1)
        self.assertEqual(f[0]["field"], "$.ingredients")
        self.assertEqual(f[0]["detail"]["values"], ["67-56-1"])
        self.assertEqual(result["review_status"], "NEEDS_REVIEW")

    def test_missing_hcode_flagged(self):
        label = clean_label()
        label["hazard_statements"] = []
        label["list_status"]["hazard_statements"] = "자료없음"
        result = run_rules(label, BASE_SOURCE_TEXT)
        self.assertIn("HCODE_IN_SOURCE_MISSING", _codes(result))

    def test_output_hcode_not_in_source_flagged(self):
        label = clean_label()
        label["hazard_statements"].append({"code": "H319", "text": "눈에 심한 자극을 일으킴"})
        result = run_rules(label, BASE_SOURCE_TEXT)
        f = [x for x in result["findings"] if x["code"] == "HCODE_NOT_IN_SOURCE"]
        self.assertEqual([x["field"] for x in f], ["$.hazard_statements[1].code"])

    def test_combined_code_split_and_checksum_filters_numbers(self):
        from app.rules.checks.source_codes import source_cas, source_hcodes
        text = "H302+H312 삼키거나 피부와 접촉하면 유해함\nEUH066 반복 노출\nH360FD\n" \
               "전화 02-123-4567, 개정 2020-01-15, 성분 7732-18-5, 7732 - 18 - 5, 틀린 7732-18-6"
        self.assertEqual(source_hcodes(text), {"H302", "H312", "H360FD"})
        self.assertEqual(source_cas(text), {"7732-18-5"})
        label = clean_label()
        label["hazard_statements"] = [{"code": "H302+H312", "text": "삼키거나 피부와 접촉하면 유해함"}]
        result = run_rules(label, "H302+H312 삼키거나 피부와 접촉하면 유해함")
        self.assertNotIn("HCODE_NOT_IN_SOURCE", _codes(result))

    def test_no_source_text_skips(self):
        label = clean_label()
        label["ingredients"] = []
        result = run_rules(label, None)
        self.assertFalse({c for c in _codes(result) if "IN_SOURCE" in c})


class TestCasCheck(unittest.TestCase):
    def test_bad_checksum(self):
        label = clean_label()
        label["ingredients"][0]["cas_number"] = "64-17-6"
        result = run_rules(label, BASE_SOURCE_TEXT)
        self.assertIn("CAS_CHECKSUM_INVALID", _codes(result))

    def test_bad_format_flags_cas_and_schema(self):
        label = clean_label()
        label["ingredients"][0]["cas_number"] = "abcde"
        result = run_rules(label, BASE_SOURCE_TEXT)
        self.assertIn("CAS_FORMAT_INVALID", _codes(result))
        self.assertIn("SCHEMA_FIELD_INVALID", _codes(result))  # src/schema.py도 형식을 검사함
        self.assertEqual(result["review_status"], "SCHEMA_ERROR")


class TestHcodeCheck(unittest.TestCase):
    def test_unknown_code_not_in_table(self):
        label = clean_label()
        label["hazard_statements"] = [{"code": "H999", "text": "표에 없는 임의의 문구"}]
        result = run_rules(label, BASE_SOURCE_TEXT)
        self.assertIn("HCODE_UNKNOWN_CODE", _codes(result))


class TestPhraseToHcodeCheck(unittest.TestCase):
    def test_mismatched_phrase(self):
        label = clean_label()
        label["hazard_statements"] = [{"code": "H225", "text": "전혀 다른 문구입니다"}]
        result = run_rules(label, BASE_SOURCE_TEXT)
        self.assertIn("HCODE_PHRASE_MISMATCH", _codes(result))

    def test_null_code_is_not_checked(self):
        label = clean_label()
        label["hazard_statements"] = [{"code": None, "text": "코드 없이 문구만 있는 정상 케이스"}]
        result = run_rules(label, BASE_SOURCE_TEXT)
        self.assertNotIn("HCODE_PHRASE_MISMATCH", _codes(result))


class TestGhsConsistencyCheck(unittest.TestCase):
    def test_hcode_does_not_match_classification_is_inconsistent(self):
        label = clean_label()
        # 인화성 액체 구분2는 H225가 기대값인데 H226만 있음 -> 모순
        label["hazard_statements"] = [{"code": "H226", "text": "인화성 액체 또는 증기"}]
        result = run_rules(label, BASE_SOURCE_TEXT)
        self.assertIn("HCODE_MISSING_FOR_CLASSIFICATION", _codes(result))
        self.assertEqual(result["review_status"], "INCONSISTENT")

    def test_signal_word_mismatch(self):
        label = clean_label()
        label["ghs_classification"] = [{"hazard_class": "인화성 액체", "category": "구분 1"}]  # 위험 기대
        label["hazard_statements"] = [{"code": "H224", "text": "극인화성 액체 및 증기"}]
        label["signal_word"]["value"] = "경고"  # 실제는 경고로 어긋남
        result = run_rules(label, BASE_SOURCE_TEXT)
        self.assertIn("SIGNAL_WORD_MISMATCH", _codes(result))
        self.assertEqual(result["review_status"], "INCONSISTENT")

    def test_classification_not_in_table_is_info_only(self):
        label = clean_label()
        label["ghs_classification"] = [{"hazard_class": "표에없는분류", "category": "구분 9"}]
        result = run_rules(label, BASE_SOURCE_TEXT)
        self.assertIn("CLASSIFICATION_NOT_IN_TABLE", _codes(result))
        info_findings = [f for f in result["findings"] if f["code"] == "CLASSIFICATION_NOT_IN_TABLE"]
        self.assertEqual(info_findings[0]["severity"], "info")


class TestGhsConsistencyPartialTable(unittest.TestCase):
    """대응표에 없는 분류가 섞이면 문서 단위 판정을 남은 분류만으로 하지 않는다."""

    def _mixed(self, signal, codes):
        label = clean_label()
        label["ghs_classification"] = [
            {"hazard_class": "인화성 액체", "category": "구분 3"},       # 표에 있음: 경고 · H226
            {"hazard_class": "표에없는분류", "category": "구분 1"},       # 표에 없음
        ]
        label["signal_word"] = {"value": signal, "source_status": "기재"}
        label["hazard_statements"] = [{"code": c, "text": "문구"} for c in codes]
        return run_rules(label, None)

    def test_stronger_signal_and_extra_code_not_flagged(self):
        result = self._mixed("위험", ["H226", "H999"])  # 위험 · H999는 표에 없는 분류 몫일 수 있음
        msgs = [f for f in result["findings"] if f["check"] == "ghs_signal_hcode_consistency" and f["severity"] == "error"]
        self.assertEqual(msgs, [])
        self.assertIn("COMPLETENESS_SKIPPED", _codes(result))

    def test_weaker_signal_still_flagged(self):
        label = clean_label()
        label["ghs_classification"] = [
            {"hazard_class": "인화성 액체", "category": "구분 2"},       # 표에 있음: 위험
            {"hazard_class": "표에없는분류", "category": "구분 1"},
        ]
        label["signal_word"] = {"value": "경고", "source_status": "기재"}
        result = run_rules(label, None)
        f = [x for x in result["findings"] if x["code"] == "SIGNAL_WORD_MISMATCH"]
        self.assertEqual(len(f), 1)
        self.assertTrue(f[0]["detail"]["at_least"])
        self.assertEqual(result["review_status"], "INCONSISTENT")

    def test_all_in_table_keeps_full_check(self):
        label = clean_label()
        label["hazard_statements"].append({"code": "H319", "text": "눈에 심한 자극을 일으킴"})
        result = run_rules(label, None)
        f = [x for x in result["findings"] if x["code"] == "HCODE_MISSING_FOR_CLASSIFICATION"]
        self.assertEqual(f[0]["detail"]["unexplained_codes"], ["H319"])
        self.assertNotIn("COMPLETENESS_SKIPPED", _codes(result))


class TestContentSumCheck(unittest.TestCase):
    def test_lower_bound_sum_exceeds_100(self):
        label = clean_label()
        label["ingredients"][0]["content"] = "60~70"
        label["ingredients"][1]["content"] = "50~60"  # 60 + 50 = 110 > 100
        result = run_rules(label, BASE_SOURCE_TEXT)
        self.assertIn("CONTENT_SUM_EXCEEDS_100", _codes(result))

    def test_unparseable_content_is_excluded_not_summed_as_zero_error(self):
        label = clean_label()
        label["ingredients"][0]["content"] = "확인불가"
        result = run_rules(label, BASE_SOURCE_TEXT)
        self.assertIn("CONTENT_SUM_UNPARSEABLE", _codes(result))
        self.assertNotIn("CONTENT_SUM_EXCEEDS_100", _codes(result))


if __name__ == "__main__":
    unittest.main()
