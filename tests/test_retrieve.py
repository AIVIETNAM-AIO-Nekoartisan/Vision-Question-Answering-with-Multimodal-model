import unittest

from online.backend.retrieve import (
    _FusableHit,
    aggregate_to_shots,
    es_hits_to_fusable,
    order_by_time,
)


def fused(file_path, shot_id, score, ts):
    return {
        "file_path": file_path,
        "score": score,
        "payload": {
            "file_path": file_path,
            "shot_id": shot_id,
            "frame": {"timestamp_seconds": ts},
            "video": {"name": shot_id.split("_")[0]},
        },
    }


class TestAggregateToShots(unittest.TestCase):
    def test_keeps_max_scoring_frame_per_shot(self):
        out = aggregate_to_shots(
            [
                fused("a.jpg", "001_shot_001", 0.9, 1.0),
                fused("b.jpg", "001_shot_001", 0.4, 2.0),
                fused("c.jpg", "001_shot_002", 0.7, 9.0),
            ],
            top_k=10,
        )
        self.assertEqual(len(out), 2)
        best = {s["shot_id"]: s for s in out}
        self.assertAlmostEqual(best["001_shot_001"]["score"], 0.9)
        self.assertEqual(best["001_shot_001"]["file_path"], "a.jpg")

    def test_respects_top_k(self):
        hits = [fused(f"{i}.jpg", f"001_shot_{i:03d}", 1.0 - i * 0.1, i) for i in range(6)]
        self.assertEqual(len(aggregate_to_shots(hits, top_k=3)), 3)

    def test_top_k_picks_highest_scores(self):
        hits = [fused(f"{i}.jpg", f"001_shot_{i:03d}", 1.0 - i * 0.1, i) for i in range(6)]
        kept = {s["shot_id"] for s in aggregate_to_shots(hits, top_k=2)}
        self.assertEqual(kept, {"001_shot_000", "001_shot_001"})

    def test_empty_input(self):
        self.assertEqual(aggregate_to_shots([], top_k=5), [])

    def test_carries_timestamp_and_payload(self):
        out = aggregate_to_shots([fused("a.jpg", "001_shot_001", 0.9, 12.5)], top_k=5)
        self.assertAlmostEqual(out[0]["timestamp"], 12.5)
        self.assertEqual(out[0]["payload"]["shot_id"], "001_shot_001")

    def test_hit_without_shot_id_is_dropped(self):
        bad = {"file_path": "x.jpg", "score": 1.0, "payload": {"file_path": "x.jpg"}}
        self.assertEqual(aggregate_to_shots([bad], top_k=5), [])


class TestOrderByTime(unittest.TestCase):
    def test_sorts_ascending_by_timestamp_not_score(self):
        shots = aggregate_to_shots(
            [
                fused("late.jpg", "001_shot_009", 0.9, 90.0),
                fused("early.jpg", "001_shot_001", 0.2, 2.0),
            ],
            top_k=10,
        )
        ordered = order_by_time(shots)
        self.assertEqual([s["file_path"] for s in ordered], ["early.jpg", "late.jpg"])

    def test_empty(self):
        self.assertEqual(order_by_time([]), [])


class TestEsAdapter(unittest.TestCase):
    """ES returns dicts; rrf_weighted_fuse needs .score/.payload attributes."""

    def keyframes(self):
        return {
            "001": [
                {
                    "file_path": "k1.jpg",
                    "shot_id": "001_shot_001",
                    "timestamp": 5.0,
                    "payload": {
                        "file_path": "k1.jpg",
                        "shot_id": "001_shot_001",
                        "frame": {"timestamp_seconds": 5.0},
                    },
                },
                {
                    "file_path": "k2.jpg",
                    "shot_id": "001_shot_002",
                    "timestamp": 50.0,
                    "payload": {
                        "file_path": "k2.jpg",
                        "shot_id": "001_shot_002",
                        "frame": {"timestamp_seconds": 50.0},
                    },
                },
            ]
        }

    def test_asr_hit_maps_to_keyframes_in_its_time_window(self):
        hits = [{"_score": 3.0, "video_id": "001", "start": 4.0, "end": 6.0}]
        out = es_hits_to_fusable(hits, self.keyframes())
        self.assertEqual([h.payload["file_path"] for h in out], ["k1.jpg"])
        self.assertTrue(all(isinstance(h, _FusableHit) for h in out))
        self.assertAlmostEqual(out[0].score, 3.0)

    def test_hit_outside_any_window_yields_nothing(self):
        hits = [{"_score": 3.0, "video_id": "001", "start": 900.0, "end": 910.0}]
        self.assertEqual(es_hits_to_fusable(hits, self.keyframes()), [])

    def test_wide_window_maps_to_several_keyframes(self):
        hits = [{"_score": 2.0, "video_id": "001", "start": 0.0, "end": 100.0}]
        out = es_hits_to_fusable(hits, self.keyframes())
        self.assertEqual(
            sorted(h.payload["file_path"] for h in out), ["k1.jpg", "k2.jpg"]
        )

    def test_unknown_video_is_ignored(self):
        hits = [{"_score": 2.0, "video_id": "999", "start": 0.0, "end": 10.0}]
        self.assertEqual(es_hits_to_fusable(hits, self.keyframes()), [])

    def test_ocr_hit_maps_by_file_path(self):
        """OCR rows name their keyframe directly, so no time window is needed."""
        hits = [{"_score": 4.0, "video_id": "001", "file_path": "k2.jpg"}]
        out = es_hits_to_fusable(hits, self.keyframes())
        self.assertEqual([h.payload["file_path"] for h in out], ["k2.jpg"])
        self.assertAlmostEqual(out[0].score, 4.0)

    def test_score_key_fallbacks(self):
        hits = [{"score": 1.5, "video_id": "001", "file_path": "k1.jpg"}]
        out = es_hits_to_fusable(hits, self.keyframes())
        self.assertAlmostEqual(out[0].score, 1.5)

    def test_output_feeds_rrf_unchanged(self):
        """The adapter's job is to be accepted by the real fusion function."""
        from online.backend.fusion import rrf_weighted_fuse

        hits = [{"_score": 3.0, "video_id": "001", "start": 0.0, "end": 100.0}]
        fusable = es_hits_to_fusable(hits, self.keyframes())
        out = rrf_weighted_fuse({"asr": fusable}, k=60, weights={"asr": 1.0})
        self.assertEqual(len(out), 2)
        self.assertIn("rrf_score", out[0])


if __name__ == "__main__":
    unittest.main()
