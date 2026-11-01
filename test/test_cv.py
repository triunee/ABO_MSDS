"""
교차검증 하네스(eval/cv.py) 회귀 테스트 — fold 나누기 · few-shot 예시 고르기 · 짝 비교 목록. GPU · 모델 없이 돈다.

  python3 test/test_cv.py -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import cv  # noqa: E402


def docs_of(spec):
    """{doc_id: (form, group)} → splits.csv 행 모양"""
    return {d: {"doc_id": d, "split": "train", "form": f, "split_group": g} for d, (f, g) in spec.items()}


class AssignFolds(unittest.TestCase):
    def test_real_splits_group_disjoint_and_balanced(self):
        docs = cv.labeled_docs()
        fold_of = cv.assign_folds(docs, 5, 42)
        self.assertEqual(set(fold_of), set(docs))
        by_group = {}
        for d, f in fold_of.items():  # 같은 제조사 그룹은 한 fold에만
            by_group.setdefault(docs[d]["split_group"], set()).add(f)
        self.assertTrue(all(len(v) == 1 for v in by_group.values()))
        sizes = [list(fold_of.values()).count(i) for i in range(5)]
        self.assertLessEqual(max(sizes) - min(sizes), 2)

    def test_deterministic(self):
        docs = cv.labeled_docs()
        self.assertEqual(cv.assign_folds(docs, 5, 42), cv.assign_folds(docs, 5, 42))

    def test_form_tie_break(self):
        # A(현행 2) · B(구서식 2)가 먼저 두 fold에 들어간 뒤, C(구서식 2)는 크기가 같으면 구서식이 없는 fold로 간다
        docs = docs_of({"A1": ("현행", "A"), "A2": ("현행", "A"), "B1": ("구서식", "B"), "B2": ("구서식", "B"),
                        "C1": ("구서식", "C"), "C2": ("구서식", "C"), "D1": ("현행", "D"), "D2": ("현행", "D")})
        f = cv.assign_folds(docs, 2, 0)
        self.assertNotEqual(f["B1"], f["C1"])


class PickFewshot(unittest.TestCase):
    def setUp(self):
        self.docs = docs_of({"KR-HANIL-002": ("현행", "HANIL"), "KR-KUMHO-002": ("현행", "KUMHO"),
                             "KR-AAA-001": ("현행", "AAA"), "KR-BBB-001": ("구서식", "BBB"),
                             "KR-CCC-001": ("현행", "CCC")})
        self.tmp = tempfile.TemporaryDirectory()
        fs = Path(self.tmp.name) / "fewshot.json"
        fs.write_text(json.dumps({"fewshot_doc_ids": ["KR-HANIL-002", "KR-KUMHO-002"]}), encoding="utf-8")
        self.patch = mock.patch.object(cv, "FEWSHOT_JSON", fs)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_keep_fixed_when_in_train(self):
        fold_of = {"KR-HANIL-002": 1, "KR-KUMHO-002": 1, "KR-AAA-001": 0, "KR-BBB-001": 0, "KR-CCC-001": 1}
        self.assertEqual(cv.pick_fewshot(self.docs, fold_of, 0), ["KR-HANIL-002", "KR-KUMHO-002"])

    def test_replace_held_out_with_same_form_train_doc(self):
        # HANIL이 평가 fold(0) → 같은 서식(현행) train 문서 중 doc_id 순 첫 문서. 구서식 BBB는 고르지 않는다
        fold_of = {"KR-HANIL-002": 0, "KR-KUMHO-002": 1, "KR-AAA-001": 0, "KR-BBB-001": 1, "KR-CCC-001": 1}
        self.assertEqual(cv.pick_fewshot(self.docs, fold_of, 0), ["KR-CCC-001", "KR-KUMHO-002"])

    def test_no_held_out_doc_as_example(self):
        docs = cv.labeled_docs()
        fold_of = cv.assign_folds(docs, 5, 42)
        for k in range(5):
            fs = cv.pick_fewshot(docs, fold_of, k)
            self.assertEqual(len(set(fs)), 2)
            self.assertTrue(all(fold_of[d] != k for d in fs))


class Comparisons(unittest.TestCase):
    def test_default_pairs(self):
        exp = {"baseline": "r1", "conditions": {"base_zs": {"type": "base_zs"}, "base_fs": {"type": "base_fs"},
                                                 "r1": {"type": "qlora"}, "r1_mlp": {"type": "qlora"}}}
        self.assertEqual(cv.comparisons(exp), [("r1", "base_fs"), ("r1_mlp", "base_fs"), ("r1", "base_zs"),
                                               ("r1_mlp", "r1")])


class ConditionCount(unittest.TestCase):
    def test_replicate_counted_apart(self):
        exp = {"conditions": {"base_fs": {"type": "base_fs"}, "r1": {"type": "qlora"},
                              "r1_s43": {"type": "qlora", "replicate_of": "r1"}}}
        self.assertEqual(cv.condition_count(exp, ["base_fs", "r1"]), "2개")
        text = cv.condition_count(exp, ["base_fs", "r1", "r1_s43"])
        self.assertTrue(text.startswith("2개 (그 밖에 seed 반복 r1_s43 = r1의 반복"))


class Consistency(unittest.TestCase):
    def test_sign_test_p(self):
        self.assertEqual(cv.sign_test_p(0, 0), 1.0)
        self.assertAlmostEqual(cv.sign_test_p(10, 3), 2 * (1 + 13 + 78 + 286) / 2 ** 13)
        self.assertEqual(cv.sign_test_p(3, 10), cv.sign_test_p(10, 3))
        self.assertEqual(cv.sign_test_p(5, 5), 1.0)

    def test_fold_and_group_tally(self):
        # 그룹 G는 문서 2건이 1승 1패 → 그룹 단위로는 무승부
        a = {"x1": {"m": 1}, "x2": {"m": 0}, "y1": {"m": 1}, "z1": {"m": 1}}
        b = {"x1": {"m": 0}, "x2": {"m": 1}, "y1": {"m": 0}, "z1": {"m": 1}}
        c = cv.consistency(a, b, list(a), {"x1": 0, "x2": 0, "y1": 1, "z1": 1},
                           {"x1": "G", "x2": "G", "y1": "H", "z1": "I"}, metric="m")
        self.assertEqual(c["folds"], {0: (0.0, 2), 1: (0.5, 2)})
        self.assertEqual(c["doc"][:3], (2, 1, 1))
        self.assertEqual(c["group"][:3], (1, 0, 2))


class RuleFieldPath(unittest.TestCase):
    def test_top_field(self):
        from eval.cv_rules import top_field
        self.assertEqual(top_field("$.ghs_classification.2.category"), "ghs_classification")
        self.assertEqual(top_field("$.hazard_statements[].text"), "hazard_statements")
        self.assertEqual(top_field("$.list_status.ingredients"), "ingredients")
        self.assertEqual(top_field("$"), "")
        self.assertEqual(top_field(None), "")


class Perturb(unittest.TestCase):
    TEXT = ("제품명 : 테스트 페인트 (백색) 대용량 제품 이름이 긴 경우\n급성 독성(흡입: 가스) : 구분 4\n"
            "○ 신호어\n- 위험\n유해·위험 문구\n작성일 2012-11-30\n물 7732-18-5 265-157-1 36~46\n")

    def test_whitespace_only(self):
        from eval import cv_perturb
        for name in cv_perturb.SETS:
            out = cv_perturb.perturb(name, "KR-X-001", self.TEXT)
            self.assertEqual(cv_perturb.squash_all(out), cv_perturb.squash_all(self.TEXT))
            self.assertEqual(out, cv_perturb.perturb(name, "KR-X-001", self.TEXT))  # seed 고정

    def test_value(self):
        from eval import cv_perturb
        out = cv_perturb.value(self.TEXT, None)
        self.assertIn("7732 - 18 - 5", out)
        self.assertIn("36 ~ 46", out)
        self.assertIn("- 위 험", out)
        self.assertIn("2012-11-30", out)       # 날짜 · EC 번호는 그대로
        self.assertIn("265-157-1", out)
        self.assertIn("유해·위험 문구", out)    # 신호어 줄이 아닌 '위험'은 그대로

    def test_layout_keeps_paren_colon_and_wraps(self):
        from eval import cv_perturb
        out = cv_perturb.layout(self.TEXT, __import__("random").Random(0))
        self.assertNotIn("(흡입:\n", out)
        self.assertTrue(all(len(ln) <= cv_perturb.WRAP or " " not in ln for ln in out.split("\n")))


if __name__ == "__main__":
    unittest.main()
