import tempfile
import unittest
from pathlib import Path

from offline.manifest import Manifest


class TestManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "manifest.db"
        self.m = Manifest(self.path)

    def tearDown(self):
        self.m.close()
        self.tmp.cleanup()

    def test_unknown_is_not_done(self):
        self.assertIsNone(self.m.status("001", "shots"))
        self.assertFalse(self.m.is_done("001", "shots"))

    def test_mark_then_done(self):
        self.m.mark("001", "shots", "done")
        self.assertTrue(self.m.is_done("001", "shots"))
        self.assertEqual(self.m.status("001", "shots"), "done")

    def test_stages_are_independent(self):
        self.m.mark("001", "shots", "done")
        self.assertFalse(self.m.is_done("001", "embed"))

    def test_mark_is_idempotent_and_overwrites(self):
        self.m.mark("001", "shots", "failed", error="boom")
        self.m.mark("001", "shots", "done")
        self.assertEqual(self.m.status("001", "shots"), "done")
        self.assertEqual(self.m.count_done("shots"), 1)

    def test_failed_is_not_done_and_keeps_error(self):
        self.m.mark("002", "asr", "failed", error="cuda oom")
        self.assertFalse(self.m.is_done("002", "asr"))
        self.assertIn("cuda oom", self.m.error("002", "asr"))

    def test_pending_excludes_done_only(self):
        self.m.mark("001", "shots", "done")
        self.m.mark("002", "shots", "failed", error="x")
        self.assertEqual(self.m.pending(["001", "002", "003"], "shots"), ["002", "003"])

    def test_pending_preserves_input_order(self):
        self.m.mark("002", "shots", "done")
        self.assertEqual(self.m.pending(["003", "002", "001"], "shots"), ["003", "001"])

    def test_survives_reopen(self):
        self.m.mark("001", "shots", "done")
        self.m.close()
        reopened = Manifest(self.path)
        self.assertTrue(reopened.is_done("001", "shots"))
        reopened.close()

    def test_watcher_query_shape(self):
        """scripts/watch_download_and_index.sh runs exactly this SQL."""
        import sqlite3
        self.m.mark("001", "index", "done")
        self.m.mark("002", "index", "done")
        self.m.mark("003", "index", "failed", error="x")
        con = sqlite3.connect(str(self.path))
        n = con.execute(
            "SELECT COUNT(DISTINCT video_id) FROM stage_state "
            "WHERE stage='index' AND status='done';"
        ).fetchone()[0]
        con.close()
        self.assertEqual(n, 2)


if __name__ == "__main__":
    unittest.main()
