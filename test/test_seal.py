"""
[양세윤] test 봉인 회귀 테스트 — 입력은 test/fixtures/(옛 라벨, 코드 테스트 전용)를 저장소 밖 임시 폴더로 복사해 쓴다.

manifest에 문서 이름이 남지 않는지, 봉인 뒤 바뀐 자료를 잡는지, 봉인은 1회만 되는지 본다.

  python3 test/test_seal.py -v
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

from eval import seal  # noqa: E402

FIX = ROOT / "test" / "fixtures"
DOCS = {"KR-NOROO-004": ("현행", "NOROO"), "KR-NOROO-005": ("구서식", "NOROO"), "KR-GSC-001": ("현행", "GSC")}


def make_test_set(tmp, docs=DOCS):
    """저장소 밖에 test 담당 자료 4종을 만든다 → seal.main 인자"""
    for sub in ("labels", "text"):
        (tmp / sub).mkdir(parents=True)
    for d in docs:
        shutil.copy(FIX / "labels" / f"{d}.json", tmp / "labels")
        shutil.copy(FIX / "text" / f"{d}.txt", tmp / "text")
    (tmp / "subset.csv").write_text("doc_id,subset\n" + "".join(f"{d},{f}\n" for d, (f, _) in docs.items()),
                                    encoding="utf-8")
    (tmp / "groups.csv").write_text("doc_id,group\n" + "".join(f"{d},{g}\n" for d, (_, g) in docs.items()),
                                    encoding="utf-8")
    return ["--label-dir", str(tmp / "labels"), "--text-dir", str(tmp / "text"),
            "--subset-csv", str(tmp / "subset.csv"), "--groups-csv", str(tmp / "groups.csv")]


class SealTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.manifest = self.tmp / "report" / "test_manifest.csv"
        self.manifest.parent.mkdir()
        self.manifest.write_text("kind,sha256\n", encoding="utf-8")  # 봉인 전(헤더만)
        self.paths = make_test_set(self.tmp / "t")
        patcher = mock.patch.object(seal, "MANIFEST", self.manifest)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_seal(self, *argv):
        buf = io.StringIO()
        code = 0
        with contextlib.redirect_stdout(buf):
            try:
                seal.main(list(argv))
            except SystemExit as e:
                code = e.code
        return code, buf.getvalue()

    def test_write_then_verify_without_names(self):
        code, out = self.run_seal("--write", *self.paths)
        self.assertEqual(code, 0, out)
        text = self.manifest.read_text(encoding="utf-8")
        for d in DOCS:  # doc_id(제조사 약칭)가 manifest · 화면 어디에도 없다
            self.assertNotIn(d, text)
            self.assertNotIn(d, out)
        self.assertIn("정답 3건 · 제조사 그룹 2개 · 서식별 구서식 1 · 현행 2", out)
        kinds = [line.split(",")[0] for line in text.splitlines()[1:]]
        self.assertEqual(kinds.count("label"), 3)
        self.assertEqual(kinds.count("text"), 3)
        for k in seal.KINDS:
            self.assertEqual(kinds.count(f"{k}_set"), 1)
        code, out = self.run_seal("--verify", *self.paths)
        self.assertEqual(code, 0, out)

    def test_write_only_once(self):
        self.run_seal("--write", *self.paths)
        before = self.manifest.read_text(encoding="utf-8")
        code, _ = self.run_seal("--write", *self.paths)
        self.assertIn("이미 봉인", str(code))
        self.assertEqual(self.manifest.read_text(encoding="utf-8"), before)

    def test_verify_catches_changed_and_renamed_files(self):
        self.run_seal("--write", *self.paths)
        label_dir = self.tmp / "t" / "labels"
        # 1) 내용 변경 → 불일치, 이름은 --show-names일 때만
        f = label_dir / "KR-GSC-001.json"
        original = f.read_bytes()
        gold = json.loads(original)
        gold["signal_word"]["value"] = "경고" if gold["signal_word"]["value"] != "경고" else "위험"
        f.write_text(json.dumps(gold, ensure_ascii=False), encoding="utf-8")
        code, out = self.run_seal("--verify", *self.paths)
        self.assertEqual(code, 1)
        self.assertIn("label: 불일치", out)
        self.assertNotIn("KR-GSC-001", out)
        _, out = self.run_seal("--verify", *self.paths, "--show-names")
        self.assertIn("KR-GSC-001", out)
        f.write_bytes(original)
        # 2) 내용은 같고 이름만 바뀜 → set 해시로 잡는다
        f.rename(label_dir / "KR-GSC-009.json")
        ok, problems, _ = seal.verify("label", label_dir)
        self.assertFalse(ok)
        self.assertIn("이름", problems[0])

    def test_require_sealed_refuses_before_seal(self):
        with self.assertRaises(SystemExit) as cm:
            seal.require_sealed("text", self.tmp / "t" / "text")
        self.assertIn("봉인 기록 없음", str(cm.exception))

    def test_write_refuses_inconsistent_sets_without_names(self):
        groups = self.tmp / "t" / "groups.csv"
        groups.write_text("doc_id,group\nKR-NOROO-004,NOROO\nKR-GSC-001,GSC\n", encoding="utf-8")  # 1건 빠짐
        code, out = self.run_seal("--write", *self.paths)
        self.assertIn("봉인 전 점검 실패", str(code))
        self.assertIn("그룹 목록의 문서가 1건 다르다", out)
        self.assertNotIn("KR-NOROO-005", out)
        self.assertEqual(self.manifest.read_text(encoding="utf-8"), "kind,sha256\n")

    def test_write_refuses_schema_error(self):
        (self.tmp / "t" / "labels" / "KR-GSC-001.json").write_text('{"product_name": 1}', encoding="utf-8")
        code, out = self.run_seal("--write", *self.paths)
        self.assertIn("봉인 전 점검 실패", str(code))
        self.assertIn("스키마를 어긴 정답 1건", out)

    def test_refuses_paths_inside_repo(self):
        argv = [a if a != str(self.tmp / "t" / "labels") else str(FIX / "labels") for a in self.paths]
        code, _ = self.run_seal("--write", *argv)
        self.assertIn("저장소 밖", str(code))



class KeyedSealTest(SealTest):
    """비밀 키 해시(--key-file) · 종류 골라 덧붙이기(--kinds) · SEAL_KEY_FILE 게이트"""

    def setUp(self):
        super().setUp()
        self.key = self.tmp / "key.txt"
        self.key.write_text("k" * 64, encoding="utf-8")

    def rows(self):
        return [tuple(line.split(",")) for line in self.manifest.read_text(encoding="utf-8").splitlines()[1:]]

    def test_keyed_text_appended_and_not_plain(self):
        self.assertEqual(self.run_seal("--write", *self.paths, "--kinds", "label,subset,group")[0], 0)
        before = self.rows()
        code, out = self.run_seal("--write", *self.paths, "--kinds", "text", "--key-file", str(self.key))
        self.assertEqual(code, 0, out)
        rows = self.rows()
        self.assertEqual(rows[:len(before)], before)          # 기존 줄은 그대로
        kinds = [k for k, _ in rows]
        self.assertNotIn("text", kinds)
        self.assertEqual(kinds.count("text_hmac"), 3)
        self.assertEqual(kinds.count("text_hmac_set"), 1)
        plain = {seal.sha256_bytes(p.read_bytes()) for p in (self.tmp / "t" / "text").glob("*.txt")}
        self.assertFalse(plain & {h for _, h in rows})        # 키 없이 만든 해시와 겹치지 않음
        code, out = self.run_seal("--verify", *self.paths, "--kinds", "text", "--key-file", str(self.key))
        self.assertEqual(code, 0, out)

    def test_wrong_key_or_changed_text_fails(self):
        self.run_seal("--write", *self.paths, "--kinds", "text", "--key-file", str(self.key))
        other = self.tmp / "other.txt"
        other.write_text("z" * 64, encoding="utf-8")
        self.assertEqual(self.run_seal("--verify", *self.paths, "--kinds", "text", "--key-file", str(other))[0], 1)
        f = next((self.tmp / "t" / "text").glob("*.txt"))
        f.write_text(f.read_text(encoding="utf-8") + " ", encoding="utf-8")
        self.assertEqual(self.run_seal("--verify", *self.paths, "--kinds", "text", "--key-file", str(self.key))[0], 1)

    def test_same_kind_twice_refused(self):
        self.run_seal("--write", *self.paths, "--kinds", "text", "--key-file", str(self.key))
        code, _ = self.run_seal("--write", *self.paths, "--kinds", "text", "--key-file", str(self.key))
        self.assertNotEqual(code, 0)

    def test_require_sealed_uses_env_key(self):
        self.run_seal("--write", *self.paths, "--kinds", "text", "--key-file", str(self.key))
        text_dir = self.paths[self.paths.index("--text-dir") + 1]
        with mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("SEAL_KEY_FILE", None)
            with self.assertRaises(SystemExit):
                seal.require_sealed("text", text_dir)
            os.environ["SEAL_KEY_FILE"] = str(self.key)
            seal.require_sealed("text", text_dir)              # 키로 대조해 통과(멈추지 않음)

    def test_short_key_refused(self):
        self.key.write_text("short", encoding="utf-8")
        code, _ = self.run_seal("--write", *self.paths, "--kinds", "text", "--key-file", str(self.key))
        self.assertNotEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
