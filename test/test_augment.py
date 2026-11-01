"""
[강덕우] 증강 코드 회귀 테스트 — 입력은 test/fixtures/(옛 라벨, 코드 테스트 전용)

  python3 test/test_augment.py -v     (폴더 이름 test가 표준 라이브러리와 겹쳐 -m unittest test.… 는 안 됨)
"""
import csv
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.prompt import PROMPTS  # noqa: E402
from pipeline.augment.check_forbidden import check, protected_counts  # noqa: E402
from pipeline.augment.mutators import MUTATORS  # noqa: E402
from pipeline.augment.renderers import RENDERERS, TARGETED_RENDERERS, cas_labels  # noqa: E402
from pipeline.guard import GuardError, load_splits, refuse_sealed, require_splits  # noqa: E402

FIX = ROOT / "test" / "fixtures"
DOCS = sorted(p.stem for p in (FIX / "labels").glob("*.json"))


def load(doc_id):
    text = (FIX / "text" / f"{doc_id}.txt").read_text(encoding="utf-8")
    label = json.loads((FIX / "labels" / f"{doc_id}.json").read_text(encoding="utf-8"))
    return text, label


class RendererTest(unittest.TestCase):
    def test_renderers_keep_values(self):
        """렌더러는 어떤 난수에서도 정답 값 · 보호 토큰을 건드리지 않는다."""
        for doc_id in DOCS:
            text, label = load(doc_id)
            for name, fn in RENDERERS.items():
                for seed in range(20):
                    new = fn(text, random.Random(seed))
                    with self.subTest(doc=doc_id, renderer=name, seed=seed):
                        self.assertEqual(check(text, new, label), [])

    def test_caslabel_changes_label_only(self):
        text = "화학물질명 관용명 CAS번호 함유량(%)\n물 Water 7732-18-5 36∼46"
        new = cas_labels(text, random.Random(0))
        self.assertIn("7732-18-5 36∼46", new)
        self.assertNotEqual(new.count("CAS"), 0)

    def test_colon_keeps_colon_inside_value(self):
        """괄호 안 콜론은 분류명 값의 일부다 — 바꾸면 hazard_class가 원문과 달라진다."""
        text = "3) 급성 독성(흡입: 가스) : 구분2"
        for seed in range(20):
            self.assertIn("(흡입: 가스)", RENDERERS["colon"](text, random.Random(seed)))


class TargetedRendererTest(unittest.TestCase):
    """r1_aug 겨냥 렌더러(--targeted): 값 · 보호 토큰은 그대로, 바꿀 곳이 없으면 None"""
    TEXT = ("2. 유해성·위험성\n가. 유해성·위험성 분류\n- 인화성 액체 : 구분2\n- 심한 눈 손상성/눈 자극성 : 구분 2A\n"
            "- 고압가스 : 액화가스\n○ 신호어\n- 위험\n○ 유해·위험 문구\n- H225 고인화성 액체 및 증기\n- H319 눈에 심한 자극을 일으킴\n"
            "○ 그림문자\n- 해당없음\n3. 구성성분의 명칭 및 함유량\n물 7732-18-5 36∼46\n해당없음")

    def test_targeted_keep_values(self):
        for doc_id in DOCS:
            text, label = load(doc_id)
            for name, fn in TARGETED_RENDERERS.items():
                for seed in range(20):
                    new = fn(text, random.Random(seed))
                    if new is not None:
                        with self.subTest(doc=doc_id, renderer=name, seed=seed):
                            self.assertEqual(check(text, new, label), [])

    def test_each_renderer(self):
        outs = {n: {fn(self.TEXT, random.Random(seed)) for seed in range(30)} for n, fn in TARGETED_RENDERERS.items()}
        self.assertTrue(any("7732 - 18 - 5" in o and "36 ∼ 46" in o and "- 위 험" in o for o in outs["valspace"]))
        self.assertTrue(any("라벨 부착 규정 없음" in o for o in outs["nonapp"]))
        self.assertTrue(all(o.endswith("\n해당없음") for o in outs["nonapp"]))  # 3항의 해당없음은 그대로
        self.assertTrue(any("Category 2A" in o and "Liquefied gas" in o for o in outs["catnote"]))
        self.assertTrue(any("구분2 ▷ 심한 눈 손상성/눈 자극성 : 구분 2A ▷ 고압가스 : 액화가스" in o for o in outs["join"]))
        self.assertTrue(any("H225 고인화성 액체 및 증기 ▷ H319" in o for o in outs["join"]))
        for n, os_ in outs.items():
            for o in os_:
                self.assertEqual(protected_counts(o), protected_counts(self.TEXT), n)

    def test_not_applicable(self):
        text = "2. 유해성·위험성\n가. 분류 : 자료없음\n3. 구성성분의 명칭 및 함유량\n성분 없음"
        for name in ("valspace", "nonapp", "catnote", "join"):
            self.assertIsNone(TARGETED_RENDERERS[name](text, random.Random(0)), name)

    def test_canonical_tokens(self):
        """같은 정답이 되는 표기는 같은 토큰으로 센다. 상태 자료없음 ↔ 해당없음은 여전히 다르다."""
        self.assertEqual(protected_counts("Category 2A"), protected_counts("구분 2A"))
        self.assertEqual(protected_counts("Liquefied gas"), protected_counts("액화가스"))
        for phrase in ("라벨 부착 규정 없음", "규정 없음", "해당 사항 없음", "분류되지 않음"):
            self.assertEqual(protected_counts(phrase), protected_counts("해당없음"), phrase)
        self.assertNotEqual(protected_counts("자료없음"), protected_counts("해당없음"))
        self.assertNotEqual(protected_counts("구분 2A"), protected_counts("구분 2"))


class CheckTest(unittest.TestCase):
    """가이드 4절의 빈틈: 신호어 · category · 부등호 · 상태값은 라벨 값 대조로 안 잡혔다."""

    def setUp(self):
        self.text, self.label = load("KR-NOROO-004")

    def assert_caught(self, new_text):
        self.assertTrue(check(self.text, new_text, self.label), new_text[:80])

    def test_signal_word_change(self):
        self.assert_caught(self.text.replace("신호어 : 경고", "신호어 : 위험"))

    def test_category_change(self):
        self.assert_caught(self.text.replace("만성 구분3", "만성 구분2"))

    def test_status_removed(self):
        self.assert_caught(self.text.replace("자료없음", "", 1))

    def test_inequality_removed(self):
        text = "함유량 : 10 이상 ~ 20 미만"
        self.assertTrue(check(text, text.replace("이상", ""), {"ingredients": []}))

    def test_noise_with_answer_token_caught(self):
        """머리글 잡음에 답 후보 문자열(위험 · 구분 N)이 끼면 거부된다."""
        self.assert_caught(self.text + "\n위험물 안내 구분 1")

    def test_protected_counts_ignore_spacing(self):
        self.assertEqual(protected_counts("위 험"), protected_counts("위험"))


class MutatorTest(unittest.TestCase):
    def test_secret(self):
        text, label = load("KR-NOROO-004")
        new_text, new_label, gone = MUTATORS["secret"](text, label, random.Random(1))
        changed = [c for c, o in zip(new_label["ingredients"], label["ingredients"]) if c != o]
        self.assertEqual(len(changed), 1)
        self.assertEqual((changed[0]["cas_number"], changed[0]["ke_number"], changed[0]["is_substitute_data"]),
                         (None, None, True))
        self.assertEqual(check(text, new_text, new_label, gone, content_mutation=True), [])
        self.assertTrue(label["ingredients"] != new_label["ingredients"], "원본 라벨을 바꾸면 안 됨(deepcopy)")

    def test_nohcode(self):
        text, label = load("KR-OCI-005")
        new_text, new_label, gone = MUTATORS["nohcode"](text, label, random.Random(1))
        self.assertTrue(all(h["code"] is None for h in new_label["hazard_statements"]))
        self.assertEqual(check(text, new_text, new_label, gone, content_mutation=True), [])

    def test_nohcode_not_applicable(self):
        text, label = load("KR-KUMHO-002")   # 해당없음 문서: 바꿀 H코드 없음
        self.assertIsNone(MUTATORS["nohcode"](text, label, random.Random(1)))


class GuardTest(unittest.TestCase):
    def test_require_splits(self):
        splits = load_splits(FIX / "splits.csv")
        require_splits(["KR-NOROO-004"], {"train"}, splits, who="t")
        with self.assertRaises(GuardError):
            require_splits(["KR-HANIL-001"], {"train"}, splits, who="t")   # val
        with self.assertRaises(GuardError):
            require_splits(["KR-NOPE-001"], {"train"}, splits, who="t")    # splits.csv에 없음

    def test_refuse_sealed(self):
        with self.assertRaises(GuardError):
            refuse_sealed("data/sealed/test.jsonl")
        with self.assertRaises(GuardError):
            refuse_sealed("eval/test")
        refuse_sealed("data/labels")


class BuildTest(unittest.TestCase):
    def run_build(self, splits_csv, out, *extra):
        cmd = [sys.executable, str(ROOT / "pipeline" / "build_jsonl.py"), "--splits", str(splits_csv),
               "--label-dir", str(FIX / "labels"), "--text-dir", str(FIX / "text"),
               "--out-dir", str(out), "--no-tokens", *extra]
        return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)

    def test_test_split_is_never_read(self):
        """splits.csv에서 test인 문서는 파일이 있어도 JSONL에 들어가지 않는다."""
        with tempfile.TemporaryDirectory() as d:
            with open(FIX / "splits.csv", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            rows[0]["split"] = "test"
            sp = Path(d) / "splits.csv"
            with open(sp, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
            r = self.run_build(sp, Path(d) / "out")
            self.assertEqual(r.returncode, 0, r.stderr)
            ids = {json.loads(l)["doc_id"] for n in ("train.jsonl", "val.jsonl")
                   for l in (Path(d) / "out" / n).read_text(encoding="utf-8").splitlines()}
            self.assertNotIn(rows[0]["doc_id"], ids)

    def test_val_fixed_across_train_options(self):
        """train 옵션을 바꿔도 val.jsonl은 한 글자도 바뀌지 않는다(r1 · r2 val 비교)."""
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a", Path(d) / "b"
            self.assertEqual(self.run_build(FIX / "splits.csv", a).returncode, 0)
            self.assertEqual(self.run_build(FIX / "splits.csv", b, "--secret", "1", "--renderers", "2").returncode, 0)
            self.assertEqual((a / "val.jsonl").read_bytes(), (b / "val.jsonl").read_bytes())
            self.assertNotEqual((a / "train.jsonl").read_bytes(), (b / "train.jsonl").read_bytes())
            rep = json.loads((a / "build_report.json").read_text(encoding="utf-8"))
            self.assertEqual(rep["checks"]["train_val_doc_overlap"], [])
            self.assertEqual(rep["checks"]["val_variants_in_train"], 0)

    def test_targeted_only_adds_examples(self):
        """--targeted는 겨냥 렌더러 예시만 더한다. 나머지 train 예시와 val.jsonl은 끈 세트와 글자까지 같다(r1 재현)."""
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a", Path(d) / "b"
            self.assertEqual(self.run_build(FIX / "splits.csv", a).returncode, 0)
            r = self.run_build(FIX / "splits.csv", b, "--targeted")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual((a / "val.jsonl").read_bytes(), (b / "val.jsonl").read_bytes())
            base = sorted((a / "train.jsonl").read_text(encoding="utf-8").splitlines())
            aug = (b / "train.jsonl").read_text(encoding="utf-8").splitlines()
            rest = sorted(l for l in aug if json.loads(l)["variant"] not in TARGETED_RENDERERS)
            self.assertEqual(rest, base)
            self.assertGreater(len(aug), len(base))
            rep = json.loads((b / "build_report.json").read_text(encoding="utf-8"))
            self.assertTrue(rep["args"]["targeted"])
            self.assertEqual([x for x in rep["splits"]["train"]["rejected"] if x["variant"] in TARGETED_RENDERERS], [])

    def test_prompt_v2_goes_to_own_folder_and_only_system_differs(self):
        """--prompt v2는 out-dir/prompt_v2/에 쓰고(v1 파일 그대로), 증강 결과는 같고 system 메시지만 다르다."""
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "out"
            self.assertEqual(self.run_build(FIX / "splits.csv", out).returncode, 0)
            v1 = {n: (out / n).read_bytes() for n in ("train.jsonl", "val.jsonl", "build_report.json")}
            r = self.run_build(FIX / "splits.csv", out, "--prompt", "v2")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual({n: (out / n).read_bytes() for n in v1}, v1)
            for n in ("train.jsonl", "val.jsonl"):
                a = [json.loads(l) for l in v1[n].decode("utf-8").splitlines()]
                b = [json.loads(l) for l in (out / "prompt_v2" / n).read_text(encoding="utf-8").splitlines()]
                self.assertEqual([(x["doc_id"], x["variant"], x["messages"][1:]) for x in a],
                                 [(x["doc_id"], x["variant"], x["messages"][1:]) for x in b])
                self.assertEqual({x["messages"][0]["content"] for x in b}, {PROMPTS["v2"]})
            rep = json.loads((out / "prompt_v2" / "build_report.json").read_text(encoding="utf-8"))
            self.assertEqual(rep["prompt"]["version"], "v2")


if __name__ == "__main__":
    unittest.main()
