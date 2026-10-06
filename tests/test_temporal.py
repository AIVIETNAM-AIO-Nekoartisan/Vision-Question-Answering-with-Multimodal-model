"""P2/P3/P4: temporal windows, interval ASR mapping, shot-level dedup."""
import unittest

from online.backend.retrieve import (
    _FusableHit,
    build_temporal_windows,
    collapse_source_to_shots,
)
from online.backend.vqa import _attach_text, _overlapping_speech


def kf(shot, pos, ts, path=None):
    return {
        "file_path": path or f"keyframes/001/{shot:04d}_{pos}.jpg",
        "shot_id": f"001_shot_{shot:03d}",
        "timestamp": ts,
        "payload": {
            "file_path": path or f"keyframes/001/{shot:04d}_{pos}.jpg",
            "shot_id": f"001_shot_{shot:03d}",
            "shot": {"number": shot, "start": ts - 1.0, "end": ts + 1.0},
            "frame": {"timestamp_seconds": ts},
            "video": {"name": "001"},
        },
    }


def ranked(shot, ts, score, srcs=("jina",)):
    k = kf(shot, 1, ts)
    return {
        "shot_id": k["shot_id"], "score": score, "timestamp": ts,
        "file_path": k["file_path"], "payload": k["payload"],
        "rrf_components": list(srcs),
    }


class TestCollapseBeforeFusion(unittest.TestCase):
    """P4: one entry per shot per source, so long shots stop winning on length."""

    def test_many_keyframes_of_one_shot_collapse_to_one(self):
        hits = [_FusableHit(score=0.1 * i, payload=kf(7, i, 10.0 + i)["payload"])
                for i in range(1, 14)]          # a 13-keyframe shot
        out = collapse_source_to_shots(hits)
        self.assertEqual(len(out), 1)

    def test_keeps_the_best_scoring_keyframe(self):
        hits = [
            _FusableHit(score=0.2, payload=kf(7, 1, 10.0)["payload"]),
            _FusableHit(score=0.9, payload=kf(7, 2, 12.0)["payload"]),
        ]
        out = collapse_source_to_shots(hits)
        self.assertAlmostEqual(out[0].score, 0.9)

    def test_distinct_shots_survive(self):
        hits = [
            _FusableHit(score=0.5, payload=kf(1, 1, 1.0)["payload"]),
            _FusableHit(score=0.4, payload=kf(2, 1, 9.0)["payload"]),
        ]
        self.assertEqual(len(collapse_source_to_shots(hits)), 2)

    def test_output_is_score_ordered(self):
        hits = [
            _FusableHit(score=0.1, payload=kf(1, 1, 1.0)["payload"]),
            _FusableHit(score=0.9, payload=kf(2, 1, 9.0)["payload"]),
        ]
        self.assertAlmostEqual(collapse_source_to_shots(hits)[0].score, 0.9)

    def test_a_long_shot_no_longer_outranks_a_short_one_by_count(self):
        """13 weak keyframes must not beat 1 strong keyframe after collapsing."""
        long_shot = [_FusableHit(score=0.30, payload=kf(7, i, 10.0 + i)["payload"])
                     for i in range(1, 14)]
        short_shot = [_FusableHit(score=0.80, payload=kf(9, 1, 60.0)["payload"])]
        out = collapse_source_to_shots(long_shot + short_shot)
        self.assertEqual(out[0].payload["shot_id"], "001_shot_009")
        self.assertEqual(len(out), 2)

    def test_hits_without_shot_id_are_dropped(self):
        self.assertEqual(collapse_source_to_shots(
            [_FusableHit(score=1.0, payload={"file_path": "x.jpg"})]), [])


class TestTemporalWindows(unittest.TestCase):
    """P2: a hit becomes a span with before/during/after frames."""

    def pool(self):
        out = []
        for shot in range(1, 12):
            base = shot * 10.0
            for pos, off in enumerate((0.0, 2.0, 4.0), start=1):
                out.append(kf(shot, pos, base + off))
        return out

    def test_window_includes_neighbour_shots(self):
        w = build_temporal_windows([ranked(5, 50.0, 0.9)], self.pool(), n_windows=1)
        self.assertEqual(w[0]["shot_numbers"], [4, 5, 6])

    def test_window_yields_several_frames_in_time_order(self):
        w = build_temporal_windows([ranked(5, 50.0, 0.9)], self.pool(), n_windows=1)
        ts = [f["timestamp"] for f in w[0]["frames"]]
        self.assertEqual(len(ts), 3)
        self.assertEqual(ts, sorted(ts))

    def test_frames_span_the_window_not_one_instant(self):
        """First and last frame must differ, or change is invisible."""
        w = build_temporal_windows([ranked(5, 50.0, 0.9)], self.pool(), n_windows=1)
        self.assertGreater(w[0]["end_time"] - w[0]["start_time"], 5.0)

    def test_adjacent_hits_merge_into_one_window(self):
        """Two hits 2s apart are the same moment; they must not take two slots."""
        hits = [ranked(5, 50.0, 0.9), ranked(5, 52.0, 0.8)]
        w = build_temporal_windows(hits, self.pool(), n_windows=3)
        self.assertEqual(len(w), 1)

    def test_merged_window_keeps_both_source_labels(self):
        hits = [ranked(5, 50.0, 0.9, ("jina",)), ranked(6, 52.0, 0.8, ("asr",))]
        w = build_temporal_windows(hits, self.pool(), n_windows=3)
        self.assertEqual(len(w), 1)
        self.assertEqual(sorted(w[0]["rrf_components"]), ["asr", "jina"])

    def test_distant_hits_stay_separate(self):
        hits = [ranked(2, 20.0, 0.9), ranked(9, 90.0, 0.8)]
        w = build_temporal_windows(hits, self.pool(), n_windows=3)
        self.assertEqual(len(w), 2)

    def test_respects_the_window_budget(self):
        hits = [ranked(s, s * 10.0, 1.0 - s * 0.01) for s in range(1, 11)]
        self.assertEqual(len(build_temporal_windows(hits, self.pool(), n_windows=3)), 3)

    def test_windows_are_returned_in_time_order(self):
        hits = [ranked(9, 90.0, 0.9), ranked(2, 20.0, 0.8)]
        w = build_temporal_windows(hits, self.pool(), n_windows=3)
        self.assertEqual([x["centre_time"] for x in w], [20.0, 90.0])

    def test_empty_inputs(self):
        self.assertEqual(build_temporal_windows([], self.pool(), n_windows=3), [])
        self.assertEqual(build_temporal_windows([ranked(5, 50.0, 0.9)], [], 3), [])


class TestOverlappingSpeech(unittest.TestCase):
    """P3a: interval overlap, not point containment on one timestamp."""

    SEGS = [
        {"text": "line at the start", "start": 10.0, "end": 12.0},
        {"text": "line in the middle", "start": 14.0, "end": 16.0},
        {"text": "far away", "start": 90.0, "end": 92.0},
    ]

    def test_speech_at_shot_start_is_kept_when_keyframe_is_at_the_end(self):
        """The case the old rule lost: shot 10-17s, keyframe at 16.5s."""
        got = _overlapping_speech(self.SEGS, 10.0, 17.0)
        self.assertIn("line at the start", got)
        self.assertIn("line in the middle", got)

    def test_point_containment_would_have_missed_it(self):
        old = [s["text"] for s in self.SEGS
               if s["start"] <= 16.5 <= s["end"]]
        self.assertEqual(old, [])          # the old rule found nothing
        self.assertTrue(_overlapping_speech(self.SEGS, 10.0, 17.0))

    def test_padding_catches_speech_just_outside(self):
        got = _overlapping_speech(self.SEGS, 17.0, 18.0, pad=1.5)
        self.assertIn("line in the middle", got)

    def test_unrelated_speech_is_excluded(self):
        self.assertNotIn("far away", _overlapping_speech(self.SEGS, 10.0, 17.0))

    def test_no_span_yields_nothing(self):
        self.assertEqual(_overlapping_speech(self.SEGS, None, None), "")

    def test_empty_segments(self):
        self.assertEqual(_overlapping_speech([], 10.0, 17.0), "")


class TestShotLevelOcr(unittest.TestCase):
    """P3b: all of a shot's text, not just the representative frame's."""

    def test_text_from_a_non_representative_frame_survives(self):
        shots = [ranked(3, 30.0, 0.9)]
        out = _attach_text(shots, [], {3: "EXIT | PLATFORM 9"})
        self.assertEqual(out[0]["ocr"], "EXIT | PLATFORM 9")

    def test_missing_shot_gives_empty_string(self):
        out = _attach_text([ranked(3, 30.0, 0.9)], [], {})
        self.assertEqual(out[0]["ocr"], "")

    def test_asr_uses_the_shot_span(self):
        segs = [{"text": "spoken early", "start": 29.0, "end": 29.5}]
        out = _attach_text([ranked(3, 30.0, 0.9)], segs, {})
        self.assertIn("spoken early", out[0]["asr"])


if __name__ == "__main__":
    unittest.main()


class TestWindowSpanIsBounded(unittest.TestCase):
    """A window must stay a moment: unbounded merging grew one to 12 shots."""

    def pool(self):
        out = []
        for shot in range(1, 30):
            base = shot * 4.0
            for pos, off in enumerate((0.0, 1.0, 2.0), start=1):
                out.append(kf(shot, pos, base + off))
        return out

    def test_many_adjacent_hits_do_not_grow_one_window_without_bound(self):
        hits = [ranked(s, s * 4.0, 1.0 - s * 0.01) for s in range(5, 20)]
        w = build_temporal_windows(hits, self.pool(), n_windows=3,
                                   max_shots_per_window=5)
        for x in w:
            self.assertLessEqual(len(x["shot_numbers"]), 5)

    def test_window_stays_near_its_centre_shot(self):
        hits = [ranked(s, s * 4.0, 1.0 - s * 0.01) for s in range(5, 20)]
        w = build_temporal_windows(hits, self.pool(), n_windows=1,
                                   max_shots_per_window=5)
        centre = w[0]["centre_shot"]
        for n in w[0]["shot_numbers"]:
            self.assertLessEqual(abs(n - centre), 3)
