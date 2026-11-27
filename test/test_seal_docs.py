"""
[양세윤] 문서 목록 봉인 회귀 테스트 — 가짜 PDF 바이트를 저장소 밖 임시 폴더에 만들어 쓴다.

  python3 test/test_seal_docs.py -v
"""
import contextlib
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import seal, seal_docs  # noqa: E402


class SealDocsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.pdf = self.tmp / "pdf"
        self.pdf.mkdir()
        for i in range(3):
            (self.pdf / f"HO-{i:03d}.pdf").write_bytes(b"%PDF-1.4 fake " + bytes([i]) * 50)
        self.key = self.tmp / "key.txt"
        self.key.write_text("a" * 64, encoding="utf-8")
        self.manifest = self.tmp / "m" / "docs_manifest.csv"
        self.args = ["--pdf-dir", str(self.pdf), "--key-file", str(self.key), "--manifest", str(self.manifest)]

    def run_main(self, *argv):
        buf = io.StringIO()
        code = 0
        with contextlib.redirect_stdout(buf):
            try:
                seal_docs.main(list(argv))
            except SystemExit as e:
                code = e.code
        return code, buf.getvalue()

    def test_write_verify_no_names_no_plain_hash(self):
        code, out = self.run_main("--write", *self.args)
        self.assertEqual(code, 0, out)
        text = self.manifest.read_text(encoding="utf-8")
        self.assertNotIn("HO-", text)
        self.assertNotIn("HO-", out)
        kinds = [line.split(",")[0] for line in text.splitlines()[1:]]
        self.assertEqual(kinds.count("pdf_hmac"), 3)
        self.assertEqual(kinds.count("pdf_hmac_set"), 1)
        plain = {seal.sha256_bytes(p.read_bytes()) for p in self.pdf.glob("*.pdf")}
        self.assertFalse(plain & {line.split(",")[1] for line in text.splitlines()[1:]})
        self.assertEqual(self.run_main("--verify", *self.args)[0], 0)

    def test_changed_added_or_wrong_key_fails(self):
        self.run_main("--write", *self.args)
        (self.pdf / "HO-009.pdf").write_bytes(b"%PDF extra")
        self.assertEqual(self.run_main("--verify", *self.args)[0], 1)
        (self.pdf / "HO-009.pdf").unlink()
        other = self.tmp / "other.txt"
        other.write_text("b" * 64, encoding="utf-8")
        args = ["--pdf-dir", str(self.pdf), "--key-file", str(other), "--manifest", str(self.manifest)]
        self.assertEqual(self.run_main("--verify", *args)[0], 1)

    def test_write_only_once(self):
        self.run_main("--write", *self.args)
        self.assertNotEqual(self.run_main("--write", *self.args)[0], 0)


if __name__ == "__main__":
    unittest.main()
