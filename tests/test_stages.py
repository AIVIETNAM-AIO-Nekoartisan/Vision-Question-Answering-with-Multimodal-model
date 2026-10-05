import unittest

import numpy as np

from offline.phash import hamming, phash
from offline.shots import predictions_to_shots, select_keyframe_positions
from offline.stages import keyframe_payload, point_id_for


class TestKeyframePayload(unittest.TestCase):
    """The nested shape is load-bearing: fusion.py keys on file_path, shot
    aggregation keys on shot_id, parsing.py reads video.name and frame.*"""

    def setUp(self):
        self.p = keyframe_payload(
            video_id="001",
            shot_number=3,
            position=1,
            frame_index=120,
            timestamp=4.0,
            file_path="keyframes/001/0003_1.webp",
            shot_start=3.5,
            shot_end=6.0,
        )

    def test_has_flat_fusion_key(self):
        self.assertEqual(self.p["file_path"], "keyframes/001/0003_1.webp")

    def test_shot_id_groups_frames(self):
        self.assertEqual(self.p["shot_id"], "001_shot_003")

    def test_nested_video_and_frame(self):
        self.assertEqual(self.p["video"]["name"], "001")
        self.assertEqual(self.p["video"]["filename"], "001.mp4")
        self.assertEqual(self.p["frame"]["index"], 120)
        self.assertAlmostEqual(self.p["frame"]["timestamp_seconds"], 4.0)

    def test_shot_bounds_present(self):
        self.assertAlmostEqual(self.p["shot"]["start"], 3.5)
        self.assertAlmostEqual(self.p["shot"]["end"], 6.0)
        self.assertEqual(self.p["shot"]["number"], 3)

    def test_passes_parsing_sanity(self):
        from core.utils.parsing import (
            extract_frame_id_from_payload,
            metadata_needs_repair,
        )
        self.assertFalse(metadata_needs_repair(self.p))
        self.assertEqual(extract_frame_id_from_payload(self.p), 120)


class TestPointId(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(point_id_for("001", 3, 120), point_id_for("001", 3, 120))

    def test_distinct_inputs_differ(self):
        self.assertNotEqual(point_id_for("001", 3, 120), point_id_for("001", 3, 121))
        self.assertNotEqual(point_id_for("001", 3, 120), point_id_for("002", 3, 120))
        self.assertNotEqual(point_id_for("001", 3, 120), point_id_for("001", 4, 120))

    def test_is_uuid(self):
        import uuid
        uuid.UUID(point_id_for("001", 3, 120))  # raises if malformed


class TestPredictionsToShots(unittest.TestCase):
    def test_single_shot_when_no_transition(self):
        preds = np.zeros(100, dtype=np.float32)
        shots = predictions_to_shots(preds, fps=25.0)
        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0]["start_frame"], 0)
        self.assertEqual(shots[0]["end_frame"], 99)

    def test_one_transition_splits_in_two(self):
        preds = np.zeros(100, dtype=np.float32)
        preds[50] = 0.9
        shots = predictions_to_shots(preds, fps=25.0, threshold=0.5)
        self.assertEqual(len(shots), 2)
        self.assertEqual(shots[0]["start_frame"], 0)
        self.assertEqual(shots[1]["end_frame"], 99)

    def test_shots_are_numbered_from_one_and_contiguous(self):
        preds = np.zeros(120, dtype=np.float32)
        preds[40] = 0.9
        preds[80] = 0.9
        shots = predictions_to_shots(preds, fps=25.0)
        self.assertEqual([s["number"] for s in shots], [1, 2, 3])
        for a, b in zip(shots, shots[1:]):
            self.assertEqual(b["start_frame"], a["end_frame"] + 1)

    def test_timestamps_follow_fps(self):
        preds = np.zeros(50, dtype=np.float32)
        shots = predictions_to_shots(preds, fps=25.0)
        self.assertAlmostEqual(shots[0]["start"], 0.0)
        self.assertAlmostEqual(shots[0]["end"], 49 / 25.0, places=5)

    def test_consecutive_high_frames_are_one_boundary(self):
        """A dissolve spans several frames but is still a single cut."""
        preds = np.zeros(100, dtype=np.float32)
        preds[50:55] = 0.9
        shots = predictions_to_shots(preds, fps=25.0)
        self.assertEqual(len(shots), 2)

    def test_empty_predictions(self):
        self.assertEqual(predictions_to_shots(np.zeros(0, dtype=np.float32), fps=25.0), [])


class TestSelectKeyframePositions(unittest.TestCase):
    def test_short_shot_gets_one_frame(self):
        """Under 1s there is nothing to sample; take the middle."""
        pos = select_keyframe_positions(start=0.0, end=0.8, fps=25.0)
        self.assertEqual(len(pos), 1)

    def test_normal_shot_gets_three(self):
        pos = select_keyframe_positions(start=0.0, end=4.0, fps=25.0)
        self.assertEqual(len(pos), 3)

    def test_long_shot_samples_every_three_seconds(self):
        pos = select_keyframe_positions(start=0.0, end=60.0, fps=25.0)
        self.assertEqual(len(pos), 20)

    def test_very_long_shot_is_capped_but_still_well_covered(self):
        """A 325s single shot really happens: videos 005 and 008 peak at 0.497
        transition probability, so TransNetV2 reports one shot for the lot."""
        pos = select_keyframe_positions(start=0.0, end=325.0, fps=30.0)
        self.assertLessEqual(len(pos), 40)
        self.assertGreaterEqual(len(pos), 30)

    def test_coverage_gap_stays_bounded_on_long_shots(self):
        """No stretch of a long shot may go unsampled for too long."""
        fps = 30.0
        for duration in (60.0, 180.0, 325.0, 600.0):
            pos = select_keyframe_positions(0.0, duration, fps)
            gaps = [(b - a) / fps for a, b in zip(pos, pos[1:])]
            self.assertLess(
                max(gaps), 20.0, f"{duration}s shot has a {max(gaps):.0f}s gap"
            )

    def test_positions_inside_shot_and_sorted(self):
        pos = select_keyframe_positions(start=10.0, end=30.0, fps=25.0)
        self.assertEqual(pos, sorted(pos))
        for f in pos:
            self.assertGreaterEqual(f, int(10.0 * 25.0))
            self.assertLessEqual(f, int(30.0 * 25.0))

    def test_positions_are_unique(self):
        pos = select_keyframe_positions(start=0.0, end=1.2, fps=25.0)
        self.assertEqual(len(pos), len(set(pos)))


class TestPhash(unittest.TestCase):
    def test_identical_images_hash_equal(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        self.assertEqual(phash(img), phash(img.copy()))

    def test_hamming_of_equal_is_zero(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        self.assertEqual(hamming(phash(img), phash(img)), 0)

    def test_near_duplicate_is_close(self):
        """Mild brightness change must stay under the dedup threshold of 4."""
        rng = np.random.default_rng(2)
        img = rng.integers(40, 200, (64, 64, 3), dtype=np.uint8)
        brighter = np.clip(img.astype(np.int16) + 12, 0, 255).astype(np.uint8)
        self.assertLessEqual(hamming(phash(img), phash(brighter)), 4)

    def test_different_images_are_far(self):
        """Two images with real frequency content must land far apart.

        Uses noise rather than flat blocks: a half-white square has only 4 of 63
        low-frequency DCT coefficients non-zero, so median thresholding sets
        barely any bits and every near-flat image hashes close to every other.
        Real imagehash behaves the same way; see test_flat_images_collapse.
        """
        rng = np.random.default_rng(7)
        a = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        b = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        self.assertGreater(hamming(phash(a), phash(b)), 4)

    def test_flat_images_collapse(self):
        """Near-featureless frames hash together, which is what dedup wants.

        Fades to black and blank frames carry no evidence, so collapsing them
        into one keyframe per shot is correct rather than a limitation.
        """
        black = np.zeros((64, 64, 3), dtype=np.uint8)
        dark = np.full((64, 64, 3), 3, dtype=np.uint8)
        self.assertLessEqual(hamming(phash(black), phash(dark)), 4)

    def test_hash_is_64_bit(self):
        rng = np.random.default_rng(3)
        img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        self.assertLess(phash(img), 1 << 64)


if __name__ == "__main__":
    unittest.main()


class TestAsrSourceSeparation(unittest.TestCase):
    """Subtitle and Whisper artifacts must not share paths or ES ids.

    The Elasticsearch _id derives from file_path, so a shared path would let
    whisper segment i overwrite subtitle segment i. The two sources produce
    different segment counts, so the index would end up a mix of both.
    """

    def _ctx(self, source):
        from pathlib import Path
        from offline.stages import Ctx
        return Ctx(
            data_root=Path("/tmp/x"),
            videos_dir=Path("/tmp/x/videos"),
            annotations_dir=Path("/tmp/x/ann"),
            transnet_weights=Path("/tmp/x/w.pth"),
            asr_source=source,
        )

    def test_subtitle_keeps_the_plain_paths(self):
        c = self._ctx("subtitle")
        self.assertEqual(c.asr_dir_name, "asr")
        self.assertEqual(c.es_asr_index_for_source, "asr_data")
        self.assertTrue(str(c.asr_json("001")).endswith("asr/001.json"))

    def test_whisper_gets_its_own_dir_and_index(self):
        c = self._ctx("whisper")
        self.assertEqual(c.asr_dir_name, "asr_whisper")
        self.assertEqual(c.es_asr_index_for_source, "asr_data_whisper")
        self.assertTrue(str(c.asr_json("001")).endswith("asr_whisper/001.json"))

    def test_the_two_sources_never_collide(self):
        a, b = self._ctx("subtitle"), self._ctx("whisper")
        self.assertNotEqual(a.asr_json("001"), b.asr_json("001"))
        self.assertNotEqual(a.es_asr_index_for_source, b.es_asr_index_for_source)
        self.assertNotEqual(a.asr_dir_name, b.asr_dir_name)
