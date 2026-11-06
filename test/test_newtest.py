"""
[양세윤] 새 test 평가(eval/newtest.py · cv/prereg_newtest.md) 회귀 테스트 — fixture(옛 라벨, 코드 테스트 전용)를
저장소 밖 임시 폴더로 복사해 쓴다. 실제 새 test 자료는 쓰지 않는다.

봉인 → 단위 12개(여기서는 축소: base 2 + 레시피당 어댑터 2) 채점 → 판정 집계를 끝까지 돌리고,
팀 test 기록(report/scores.csv · report/test_manifest.csv)을 건드리지 않는지, 문서 이름이 저장소 쪽 결과에 남지 않는지 본다.

  python3 test/test_newtest.py -v
"""
import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "test"))

from eval import infer, newtest, score, seal  # noqa: E402
from test_seal import make_test_set  # noqa: E402

FIX = ROOT / "test" / "fixtures"
DOCS = {"KR-NOROO-004": ("현행", "NOROO"), "KR-NOROO-005": ("구서식", "NOROO"), "KR-GSC-001": ("현행", "GSC"),
        "KR-KUMHO-002": ("현행", "KUMHO"), "KR-OCI-005": ("현행", "OCI")}
RECIPES = {"r1": ["r1_f0", "r1_f1"], "r1_aug": ["r1_aug_f0", "r1_aug_f1"]}


def label(doc_id):
    return json.loads((FIX / "labels" / f"{doc_id}.json").read_text(encoding="utf-8"))


def quiet(fn, argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(argv)
    return buf.getvalue()


class NewTestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.root = self.tmp / "nt"
        self.set_argv = make_test_set(self.root, DOCS)
        self.manifest = self.tmp / "test_manifest.csv"
        self.exp = self.tmp / "experiment.json"
        units = [u for us in RECIPES.values() for u in us]
        self.exp.write_text(json.dumps({
            "prereg": "cv/prereg_newtest.md", "prereg_commit": "test", "max_new_tokens": 2048, "prompt_version": "v1",
            "gpu": "test", "allow_resume": True, "recipes": RECIPES,
            "adapter_paths": {u: f"cv/runs/x/{u}/adapter" for u in units},
            "adapters": {u: f"hash-{u}" for u in units}}), encoding="utf-8")
        self.nt_dir = self.tmp / "nt_results"  # 저장소의 cv/newtest 대신
        self.team_scores = self.tmp / "team_scores.csv"  # 팀 test 점수 파일 자리 — 생기면 안 됨
        self.team_manifest = self.tmp / "team_manifest.csv"
        for target, attr, value in ((newtest, "NT_DIR", self.nt_dir), (score, "SCORES_CSV", self.team_scores),
                                    (seal, "MANIFEST", self.team_manifest)):
            p = mock.patch.object(target, attr, value)
            p.start()
            self.addCleanup(p.stop)
        # 예측: base 둘은 제품명을 모두 틀림, r1 어댑터는 모두 맞음, r1_aug_f1은 한 문서만 틀림
        for unit, cond, _ in newtest.units(newtest.load_exp(self.exp)):
            d = self.root / "out" / unit / cond / "test"
            d.mkdir(parents=True)
            for doc_id in DOCS:
                pred = label(doc_id)
                if unit in newtest.BASES or (unit == "r1_aug_f1" and doc_id == "KR-GSC-001"):
                    pred["product_name"]["value"] = "다른 제품"
                (d / f"{doc_id}.json").write_text(json.dumps(pred, ensure_ascii=False), encoding="utf-8")

    def nt(self, *argv):
        return quiet(newtest.main, ["--experiment", str(self.exp), "--manifest", str(self.manifest), *argv])

    def score_all(self):
        p = newtest.paths(self.root)
        for unit, cond, adapter in newtest.units(newtest.load_exp(self.exp)):
            _, cmd = newtest.commands(None, p, unit, cond, adapter, self.exp, self.manifest)
            quiet(score.main, cmd[2:])

    def seal_set(self):
        quiet(seal.main, ["--write", "--manifest", str(self.manifest), *self.set_argv])

    def test_end_to_end(self):
        self.seal_set()
        self.assertFalse(self.team_manifest.exists())
        self.score_all()
        self.assertFalse(self.team_scores.exists())  # 팀 test 점수 파일에 섞이지 않음
        self.assertTrue((self.root / "out" / "r1_f0" / "_scores.csv").exists())  # 저장소 밖 단위 폴더에
        out = self.nt("report", "--root", str(self.root), "--n-boot", "300")
        self.assertIn("주 판정(r1 vs base_fs): 학습에 쓰지 않은 제조사 4곳 5건에서 학습의 이점이 확인됐다", out)
        self.assertIn("레시피 판정(r1_aug vs r1): 새 문서에서 개선을 확인하지 못했다", out)
        report = (self.nt_dir / "report.md").read_text(encoding="utf-8")
        results = report + (self.nt_dir / "scores.csv").read_text(encoding="utf-8") + \
            (self.nt_dir / "paired.csv").read_text(encoding="utf-8") + out
        for doc_id in DOCS:  # 저장소 쪽 결과 · 화면에는 문서 이름이 없다
            self.assertNotIn(doc_id, results)
            self.assertNotIn(doc_id.split("-")[1], results)
        self.assertIn("fold 어댑터 5개 사이에서 주지표가 갈린 문서: r1 0건 · r1_aug 1건", report)
        main = next(x for x in (self.nt_dir / "paired.csv").read_text(encoding="utf-8").splitlines()
                    if x.startswith("r1,base_fs,all,doc_exact_ext,"))
        self.assertIn("r1,base_fs,all,doc_exact_ext,5,4,5,0,0,1.0000,1.0000,1.0000", main)
        per_doc = (self.root / "out" / "_newtest_per_doc.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(per_doc), 5)  # 문서별 결과는 저장소 밖에만

    def test_report_refuses_changed_scorer(self):
        self.seal_set()
        self.score_all()
        first = self.root / "out" / "r1_f0" / "qlora_final" / "test" / score.FIRST_FILE
        lines = first.read_text(encoding="utf-8").splitlines()
        meta = json.loads(lines[0]) | {"scorer_sha256": "changed"}
        first.chmod(0o644)
        first.write_text("\n".join([json.dumps(meta)] + lines[1:]) + "\n", encoding="utf-8")
        with self.assertRaises(SystemExit) as cm:
            self.nt("report", "--root", str(self.root), "--n-boot", "100")
        self.assertIn("채점기", str(cm.exception))

    def test_report_refuses_unfinished_and_unsealed(self):
        with self.assertRaises(SystemExit) as cm:
            self.nt("report", "--root", str(self.root))
        self.assertIn("봉인", str(cm.exception))
        self.seal_set()
        with self.assertRaises(SystemExit) as cm:
            self.nt("report", "--root", str(self.root))
        self.assertIn("채점이 끝나지 않은 단위", str(cm.exception))

    def test_run_needs_seal_and_dry_run_lists_commands(self):
        with self.assertRaises(SystemExit) as cm:
            self.nt("run", "--root", str(self.root))
        self.assertIn("봉인", str(cm.exception))
        shutil.rmtree(self.root / "out" / "r1_f0")  # 이 단위만 추론부터
        out = self.nt("run", "--root", str(self.root), "--dry-run")
        self.assertIn("단위 6개 중 남은 것 6개", out)
        self.assertEqual(out.count("] infer:"), 1)  # 출력이 모두 있는 단위는 추론을 건너뛴다
        self.assertIn("[r1_f0] infer:", out)
        self.assertIn("--adapter " + str(ROOT / "cv/runs/x/r1_f0/adapter"), out)
        self.assertEqual(out.count(f"--experiment {self.exp} --manifest {self.manifest}"), 7)  # score 6 + infer 1

    def test_root_inside_repo_refused(self):
        with self.assertRaises(SystemExit) as cm:
            self.nt("status", "--root", str(ROOT / "data"))
        self.assertIn("저장소 밖", str(cm.exception))


class ScoreAndInferOptions(unittest.TestCase):
    def test_score_other_set_needs_manifest_and_outside_scores(self):
        base = ["--condition", "base_zs", "--split", "test", "--allow-test", "--experiment", "/tmp/x.json"]
        for argv, msg in ((base, "--manifest"),
                          (base + ["--manifest", "/tmp/m.csv", "--scores-csv", str(ROOT / "report" / "x.csv")],
                           "저장소 밖"),
                          (["--condition", "base_zs", "--split", "val", "--manifest", "/tmp/m.csv"], "test 전용")):
            with self.subTest(msg=msg), self.assertRaises(SystemExit), \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                score.main(argv)
            self.assertIn(msg, err.getvalue())

    def test_experiment_with_adapter_list(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        exp = tmp / "e.json"
        exp.write_text(json.dumps({"max_new_tokens": 2048, "prompt_version": "v1", "adapters": {"a": "h1", "b": "h2"}}),
                       encoding="utf-8")
        self.assertEqual(infer.allowed_adapters(infer.load_experiment(exp)), {"h1": "a", "h2": "b"})
        exp.write_text(json.dumps({"max_new_tokens": 2048, "prompt_version": "v1"}), encoding="utf-8")
        with self.assertRaises(SystemExit) as cm:
            infer.load_experiment(exp)
        self.assertIn("adapter_sha256 또는 adapters", str(cm.exception))
        # 팀 test 고정 파일은 지금처럼 adapter_sha256 하나
        team = infer.load_experiment()
        self.assertEqual(list(infer.allowed_adapters(team)), [team["adapter_sha256"]])

    def test_newtest_experiment_matches_prereg(self):
        """cv/newtest/experiment.json의 어댑터 해시가 사전 등록 1절 표와 같다(커밋된 두 파일끼리 대조)."""
        exp = newtest.load_exp()
        prereg = (ROOT / exp["prereg"]).read_text(encoding="utf-8")
        for unit, h in exp["adapters"].items():
            recipe, fold = unit.rsplit("_f", 1)
            self.assertIn(f"| {recipe} | {fold} | `{h}` |", prereg)
        self.assertEqual(len(exp["adapters"]), 10)
        self.assertEqual(exp["max_new_tokens"], json.loads((ROOT / "eval" / "experiment.json").read_text())["max_new_tokens"])


if __name__ == "__main__":
    unittest.main()
