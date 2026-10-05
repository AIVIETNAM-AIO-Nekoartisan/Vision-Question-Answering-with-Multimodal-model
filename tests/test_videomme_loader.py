import unittest

from data.videomme import (
    Segment,
    group_words_into_segments,
    is_dev,
    parse_options,
)


def w(text, start, end):
    return {"text": text, "start_time": start, "end_time": end}


class TestGroupWords(unittest.TestCase):
    def test_empty_input(self):
        self.assertEqual(group_words_into_segments([]), [])

    def test_single_word(self):
        out = group_words_into_segments([w("Hi,", 1.0, 1.4)])
        self.assertEqual(out, [Segment(text="Hi,", start=1.0, end=1.4)])

    def test_words_close_together_join(self):
        out = group_words_into_segments([w("Hi", 1.0, 1.2), w("there", 1.25, 1.6)])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].text, "Hi there")
        self.assertAlmostEqual(out[0].start, 1.0)
        self.assertAlmostEqual(out[0].end, 1.6)

    def test_long_gap_splits(self):
        """A pause longer than max_gap starts a new segment."""
        out = group_words_into_segments(
            [w("first", 1.0, 1.2), w("second", 5.0, 5.3)], max_gap=0.6
        )
        self.assertEqual([s.text for s in out], ["first", "second"])

    def test_duration_cap_splits(self):
        """Continuous speech is still cut at max_duration."""
        words = [w(f"w{i}", i * 0.5, i * 0.5 + 0.4) for i in range(60)]
        out = group_words_into_segments(words, max_gap=0.6, max_duration=15.0)
        self.assertGreater(len(out), 1)
        for seg in out:
            self.assertLessEqual(seg.end - seg.start, 15.0 + 1e-6)

    def test_gap_measured_between_words_not_from_segment_start(self):
        """Three words with small gaps stay together even past 2x max_gap total."""
        out = group_words_into_segments(
            [w("a", 0.0, 0.2), w("b", 0.7, 0.9), w("c", 1.4, 1.6)], max_gap=0.6
        )
        self.assertEqual(len(out), 1)

    def test_blank_words_are_skipped(self):
        out = group_words_into_segments(
            [w("a", 0.0, 0.2), w("   ", 0.3, 0.4), w("b", 0.5, 0.7)]
        )
        self.assertEqual(out[0].text, "a b")

    def test_no_segment_is_empty_or_inverted(self):
        words = [w(f"w{i}", i * 0.9, i * 0.9 + 0.3) for i in range(40)]
        for seg in group_words_into_segments(words):
            self.assertTrue(seg.text.strip())
            self.assertLessEqual(seg.start, seg.end)


class TestParseOptions(unittest.TestCase):
    def test_eight_options(self):
        text = ("A. Malaysian. B. British. C. Singaporean. D. German. "
                "E. Canadian. F. Chinese. G. American. H. Cannot be determined.")
        out = parse_options(text)
        self.assertEqual(len(out), 8)
        self.assertTrue(out[0].startswith("A."))
        self.assertTrue(out[7].startswith("H."))

    def test_labels_are_preserved(self):
        out = parse_options("A. Yes. B. No.")
        self.assertEqual(out, ["A. Yes.", "B. No."])

    def test_empty(self):
        self.assertEqual(parse_options(""), [])


class TestDevSplit(unittest.TestCase):
    def test_dev_boundary(self):
        self.assertTrue(is_dev("001"))
        self.assertTrue(is_dev("050"))
        self.assertFalse(is_dev("051"))
        self.assertFalse(is_dev("200"))


if __name__ == "__main__":
    unittest.main()
