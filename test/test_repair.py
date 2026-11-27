"""
[양세윤] 출력 보정(core/repair.py) 회귀 테스트 — 규칙마다 합성 예시, 그리고 정답 라벨(data/labels · train · val)에
적용하면 바뀌는 것이 없는지 본다. test 자료는 쓰지 않는다.

  python3 test/test_repair.py -v
"""
import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.repair import REPAIRS, repair_output  # noqa: E402


def label(**kw):
    base = {"hazard_statements": [], "ghs_classification": []}
    base.update(kw)
    return base


class RepairV1Test(unittest.TestCase):
    def test_r1_leading_code_moves_to_code(self):
        out, used = repair_output(label(hazard_statements=[{"code": None, "text": "H226 인화성 액체 및 증기"},
                                                           {"code": None, "text": "H302+H312: 삼키거나 피부와 접촉하면 유해함"}]))
        self.assertEqual(out["hazard_statements"][0], {"code": "H226", "text": "인화성 액체 및 증기"})
        self.assertEqual(out["hazard_statements"][1], {"code": "H302+H312", "text": "삼키거나 피부와 접촉하면 유해함"})
        self.assertEqual(used, ["R1", "R1"])

    def test_r2_code_from_one_source_line_only(self):
        src = "○ 유해 위험 문구 :\nH302 삼키면 유해함\nH412 장기적 영향에 의해 수생생물에게 유해함\n"
        out, used = repair_output(label(hazard_statements=[{"code": None, "text": "삼키면 유해함"},
                                                           {"code": None, "text": "원문에 코드 없는 문구"}]), src)
        self.assertEqual(out["hazard_statements"][0]["code"], "H302")
        self.assertIsNone(out["hazard_statements"][1]["code"])
        self.assertEqual(used, ["R2"])
        two = "H302 삼키면 유해함\nH303 삼키면 유해함\n"  # 같은 문구에 코드가 둘이면 고르지 않는다
        self.assertIsNone(repair_output(label(hazard_statements=[{"code": None, "text": "삼키면 유해함"}]), two)[0]
                          ["hazard_statements"][0]["code"])

    def test_r3_broken_code_to_null_but_valid_kept(self):
        out, used = repair_output(label(hazard_statements=[{"code": "17", "text": "알레르기성 피부 반응"},
                                                           {"code": "H360FD", "text": "x"},
                                                           {"code": "H302+H332", "text": "y"}]))
        self.assertEqual([s["code"] for s in out["hazard_statements"]], [None, "H360FD", "H302+H332"])
        self.assertEqual(used, ["R3"])

    def test_r4_group_head_and_tail_category(self):
        out, used = repair_output(label(ghs_classification=[
            {"hazard_class": "건강 유해성 피부 과민성", "category": "구분 1"},
            {"hazard_class": "급성독성물질 경구 구분", "category": "구분 4"},
            {"hazard_class": "인화성 액체", "category": "구분 2"}]))
        self.assertEqual([c["hazard_class"] for c in out["ghs_classification"]],
                         ["피부 과민성", "급성독성물질 경구", "인화성 액체"])
        self.assertEqual(used, ["R4", "R4"])

    def test_r5_split_routes_only_when_nothing_else(self):
        out, used = repair_output(label(ghs_classification=[{"hazard_class": "급성 독성-경피, 경구", "category": "구분 4"}]))
        self.assertEqual(out["ghs_classification"], [{"hazard_class": "급성 독성(경피)", "category": "구분 4"},
                                                     {"hazard_class": "급성 독성(경구)", "category": "구분 4"}])
        self.assertEqual(used, ["R5"])
        keep = [{"hazard_class": "급성 독성(경구)", "category": "구분 4"},
                {"hazard_class": "급성 독성 경구, 흡입(증기)", "category": "구분 3"}]  # 경로 말고 다른 말이 있으면 펼치지 않음
        self.assertEqual(repair_output(label(ghs_classification=keep))[0]["ghs_classification"], keep)

    def test_input_not_mutated_and_non_dict_passthrough(self):
        pred = label(hazard_statements=[{"code": "17", "text": "x"}])
        before = copy.deepcopy(pred)
        repair_output(pred)
        self.assertEqual(pred, before)
        self.assertEqual(repair_output(None), (None, []))
        with self.assertRaises(ValueError):
            repair_output(pred, version="v0")

    def test_idempotent(self):
        pred = label(hazard_statements=[{"code": None, "text": "H226 인화성"}],
                     ghs_classification=[{"hazard_class": "급성 독성-경피, 경구", "category": "구분 4"}])
        once, _ = repair_output(pred)
        twice, used = repair_output(once)
        self.assertEqual(once, twice)
        self.assertEqual(used, [])

    def test_gold_labels_unchanged(self):
        """정답 라벨(train · val)은 이미 규칙대로라서 보정해도 바뀌지 않아야 한다."""
        labels = sorted((ROOT / "data" / "labels").glob("*.json"))
        self.assertTrue(labels)
        for v in REPAIRS:
            for f in labels:
                gold = json.loads(f.read_text(encoding="utf-8"))
                t = ROOT / "data" / "text" / f"{f.stem}.txt"
                src = t.read_text(encoding="utf-8") if t.exists() else None
                out, used = repair_output(gold, src, v)
                self.assertEqual(out, gold, f"{v} {f.stem} {used}")


if __name__ == "__main__":
    unittest.main()
