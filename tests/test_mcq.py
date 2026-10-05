import unittest

from online.backend.vqa import build_mcq_prompt, condense_transcript, parse_letter


class TestParseLetter(unittest.TestCase):
    def test_bare_letter(self):
        self.assertEqual(parse_letter("C", 8), ("C", False))

    def test_letter_with_period(self):
        self.assertEqual(parse_letter("B.", 8), ("B", False))

    def test_sentence_form(self):
        self.assertEqual(parse_letter("The answer is C.", 8), ("C", False))

    def test_lowercase_is_normalised(self):
        self.assertEqual(parse_letter("f", 8), ("F", False))

    def test_leading_whitespace_and_newlines(self):
        self.assertEqual(parse_letter("\n\n  D\n", 8), ("D", False))

    def test_parenthesised(self):
        self.assertEqual(parse_letter("(E)", 8), ("E", False))

    def test_letter_beyond_option_count_is_rejected(self):
        """With 4 options, 'G' cannot be right — treat as unparsed."""
        self.assertEqual(parse_letter("G", 4), (None, True))

    def test_empty_output_is_unparsed(self):
        self.assertEqual(parse_letter("", 8), (None, True))

    def test_none_is_unparsed(self):
        self.assertEqual(parse_letter(None, 8), (None, True))

    def test_prose_without_a_letter_is_unparsed(self):
        self.assertEqual(
            parse_letter("I cannot tell from these frames.", 8), (None, True)
        )

    def test_first_standalone_letter_wins(self):
        self.assertEqual(parse_letter("A or B? I pick B", 8), ("A", False))

    def test_word_starting_with_letter_is_not_matched(self):
        """'Alice' must not read as 'A'."""
        self.assertEqual(parse_letter("Alice is the mother", 8), (None, True))

    def test_answer_prefix_word_is_not_matched(self):
        """The word 'Answer' begins with A but is not a choice."""
        self.assertEqual(parse_letter("Answer: D", 8), ("D", False))

    def test_option_count_is_clamped(self):
        """n_options outside 1..8 must not widen or empty the allowed set."""
        self.assertEqual(parse_letter("H", 99), ("H", False))
        self.assertEqual(parse_letter("A", 0), ("A", False))


class TestCondenseTranscript(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(condense_transcript([], max_chars=100), "")

    def test_under_budget_keeps_everything(self):
        segs = [{"text": "hello", "start": 0.0, "end": 1.0},
                {"text": "world", "start": 1.0, "end": 2.0}]
        out = condense_transcript(segs, max_chars=200)
        self.assertIn("hello", out)
        self.assertIn("world", out)

    def test_respects_budget(self):
        segs = [{"text": f"segment number {i}", "start": i, "end": i + 1}
                for i in range(200)]
        out = condense_transcript(segs, max_chars=300)
        self.assertLessEqual(len(out), 300)

    def test_keeps_chronological_order_when_sampling(self):
        import re

        segs = [{"text": f"s{i}", "start": float(i), "end": i + 1.0}
                for i in range(100)]
        out = condense_transcript(segs, max_chars=200)
        # Convert each [MM:SS] to total seconds; reading the SS field alone
        # would wrap at 60 and look unsorted even when the order is right.
        stamps = [
            int(mm) * 60 + int(ss)
            for mm, ss in re.findall(r"\[(\d{2}):(\d{2})\]", out)
        ]
        self.assertGreater(len(stamps), 1)
        self.assertEqual(stamps, sorted(stamps))

    def test_samples_across_the_whole_video_not_just_the_start(self):
        """A question about the ending must not lose the ending to truncation."""
        import re

        segs = [{"text": f"s{i}", "start": float(i * 10), "end": i * 10 + 5.0}
                for i in range(100)]
        out = condense_transcript(segs, max_chars=300)
        stamps = [
            int(mm) * 60 + int(ss)
            for mm, ss in re.findall(r"\[(\d{2}):(\d{2})\]", out)
        ]
        self.assertGreater(max(stamps), 500)


class TestBuildMcqPrompt(unittest.TestCase):
    def setUp(self):
        self.shots = [
            {"shot_id": "001_shot_001", "timestamp": 5.0, "file_path": "a.jpg",
             "asr": "she opens the door", "ocr": "EXIT"},
            {"shot_id": "001_shot_009", "timestamp": 95.0, "file_path": "b.jpg",
             "asr": "", "ocr": ""},
        ]
        self.options = ["A. Yes.", "B. No.", "C. Cannot be determined."]

    def test_contains_question_and_all_options(self):
        p = build_mcq_prompt("Did she leave?", self.options, self.shots, "", 0)
        self.assertIn("Did she leave?", p)
        for opt in self.options:
            self.assertIn(opt, p)

    def test_instructs_single_letter(self):
        p = build_mcq_prompt("Q?", self.options, self.shots, "", 0)
        self.assertIn("single letter", p.lower())

    def test_timestamps_are_rendered_mmss(self):
        p = build_mcq_prompt("Q?", self.options, self.shots, "", 0)
        self.assertIn("[00:05]", p)
        self.assertIn("[01:35]", p)

    def test_evidence_appears_in_timestamp_order(self):
        p = build_mcq_prompt("Q?", self.options, self.shots, "", 0)
        self.assertLess(p.index("[00:05]"), p.index("[01:35]"))

    def test_asr_and_ocr_are_included_when_present(self):
        p = build_mcq_prompt("Q?", self.options, self.shots, "", 0)
        self.assertIn("she opens the door", p)
        self.assertIn("EXIT", p)

    def test_global_context_is_labelled_when_provided(self):
        p = build_mcq_prompt("Q?", self.options, self.shots, "a summary", 8)
        self.assertIn("a summary", p)
        self.assertIn("8", p)

    def test_works_with_no_shots(self):
        p = build_mcq_prompt("Q?", self.options, [], "", 0)
        self.assertIn("Q?", p)


if __name__ == "__main__":
    unittest.main()
