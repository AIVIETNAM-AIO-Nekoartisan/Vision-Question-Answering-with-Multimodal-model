"""Skips unless the backend is actually up, matching the AIC convention."""
import json
import os
import unittest
import urllib.error
import urllib.request

API = os.getenv("API_URL", "http://localhost:8000")


def _health():
    try:
        with urllib.request.urlopen(f"{API}/health", timeout=3) as r:
            return json.loads(r.read()) if r.status == 200 else None
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


_HEALTH = _health()


def _post(path, body, timeout=120):
    req = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


@unittest.skipUnless(_HEALTH, f"backend not running at {API}")
class TestHealth(unittest.TestCase):
    def test_reports_stores_and_models(self):
        self.assertIn("qdrant", _HEALTH)
        self.assertIn("elasticsearch", _HEALTH)
        self.assertIn("models", _HEALTH)

    def test_index_is_not_empty(self):
        """A green health check over an empty index would hide a broken ingest."""
        qdrant = _HEALTH.get("qdrant") or {}
        self.assertGreater(qdrant.get("points", 0), 0)


@unittest.skipUnless(_HEALTH, f"backend not running at {API}")
class TestSearch(unittest.TestCase):
    QUERY = "a person talking to someone"

    def test_kis_returns_shots_with_required_fields(self):
        out = _post("/search/kis", {"query": self.QUERY, "top_k": 5})
        self.assertIsInstance(out.get("results"), list)
        for item in out["results"]:
            self.assertIn("shot_id", item)
            self.assertIn("file_path", item)
            self.assertIn("timestamp", item)

    def test_video_filter_restricts_to_one_video(self):
        out = _post(
            "/search/kis", {"query": self.QUERY, "top_k": 5, "video_id": "001"}
        )
        self.assertTrue(out["results"], "expected hits inside video 001")
        for item in out["results"]:
            self.assertEqual(item["payload"]["video"]["name"], "001")

    def test_results_are_time_ordered(self):
        out = _post("/search/kis", {"query": self.QUERY, "top_k": 8})
        stamps = [i["timestamp"] for i in out["results"]]
        self.assertEqual(stamps, sorted(stamps))

    def test_top_k_is_respected(self):
        out = _post("/search/kis", {"query": self.QUERY, "top_k": 3})
        self.assertLessEqual(len(out["results"]), 3)

    def test_audio_only_search_runs(self):
        out = _post("/search/audio", {"query": self.QUERY, "top_k": 5})
        self.assertIn("results", out)


@unittest.skipUnless(_HEALTH, f"backend not running at {API}")
class TestFiles(unittest.TestCase):
    def test_serves_a_keyframe(self):
        out = _post("/search/kis", {"query": "a person", "top_k": 1})
        if not out["results"]:
            self.skipTest("no hits to fetch")
        path = out["results"][0]["file_path"]
        with urllib.request.urlopen(
            f"{API}/files?p={urllib.request.quote(path)}", timeout=20
        ) as r:
            self.assertEqual(r.status, 200)
            self.assertTrue(r.read(4))

    def test_traversal_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(f"{API}/files?p=../../../../etc/passwd", timeout=10)
        self.assertIn(cm.exception.code, (403, 404))


@unittest.skipUnless(
    _HEALTH and (_HEALTH.get("models", {}).get("qwen") == "ready"),
    "VLM not loaded",
)
class TestVQA(unittest.TestCase):
    def test_answers_with_a_single_letter(self):
        out = _post("/vqa/answer", {"question_id": "001-1"}, timeout=300)
        self.assertEqual(out["video_id"], "001")
        self.assertFalse(out["unparsed"], f"could not parse: {out['raw']!r}")
        self.assertIn(out["answer"], list("ABCDEFGH"))
        self.assertIn(out["gold"], list("ABCDEFGH"))


if __name__ == "__main__":
    unittest.main()
