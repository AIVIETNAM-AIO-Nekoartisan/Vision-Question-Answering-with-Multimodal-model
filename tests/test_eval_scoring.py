import unittest

from data.videomme import Question
from eval.configs import CONFIGS, get_config
from eval.run_eval import score_group, summarise


def q(qid, answer, level="1", group_type="relevance", head="Frame-Only"):
    return Question(
        video_id=qid.split("-")[0],
        question_id=qid,
        question="?",
        options=[f"{c}. x" for c in "ABCDEFGH"],
        answer=answer,
        level=level,
        group_type=group_type,
        second_head=head,
        third_head="t",
    )


class TestGroupScoring(unittest.TestCase):
    def test_relevance_scores_each_question(self):
        qs = [q("001-1", "A"), q("001-2", "B"), q("001-3", "C")]
        preds = {"001-1": "A", "001-2": "X", "001-3": "C"}
        self.assertAlmostEqual(score_group(qs, preds, "relevance"), 2 / 3)

    def test_logic_truncates_after_first_error(self):
        """Wrong on #2 zeroes #3 and #4 even though they match."""
        qs = [
            q("001-1", "A", group_type="logic"),
            q("001-2", "B", group_type="logic"),
            q("001-3", "C", group_type="logic"),
            q("001-4", "D", group_type="logic"),
        ]
        preds = {"001-1": "A", "001-2": "X", "001-3": "C", "001-4": "D"}
        self.assertAlmostEqual(score_group(qs, preds, "logic"), 1 / 4)

    def test_logic_all_correct_is_full_marks(self):
        qs = [q(f"001-{i}", "A", group_type="logic") for i in range(1, 5)]
        preds = {f"001-{i}": "A" for i in range(1, 5)}
        self.assertAlmostEqual(score_group(qs, preds, "logic"), 1.0)

    def test_logic_first_wrong_scores_zero(self):
        qs = [q(f"001-{i}", "A", group_type="logic") for i in range(1, 5)]
        preds = {"001-1": "X", "001-2": "A", "001-3": "A", "001-4": "A"}
        self.assertAlmostEqual(score_group(qs, preds, "logic"), 0.0)

    def test_logic_orders_by_question_id_not_dict_order(self):
        """Truncation depends on sequence, so ordering must not come from preds."""
        qs = [
            q("001-2", "B", group_type="logic"),
            q("001-1", "A", group_type="logic"),
        ]
        preds = {"001-2": "B", "001-1": "X"}
        self.assertAlmostEqual(score_group(qs, preds, "logic"), 0.0)

    def test_unparsed_counts_as_wrong(self):
        qs = [q("001-1", "A"), q("001-2", "B")]
        preds = {"001-1": None, "001-2": "B"}
        self.assertAlmostEqual(score_group(qs, preds, "relevance"), 0.5)

    def test_empty_group(self):
        self.assertAlmostEqual(score_group([], {}, "relevance"), 0.0)


class TestSummarise(unittest.TestCase):
    def test_reports_n_accuracy_and_unparsed(self):
        qs = [q("001-1", "A", level="1"), q("001-2", "B", level="2")]
        preds = {"001-1": "A", "001-2": None}
        out = summarise(qs, preds)
        self.assertEqual(out["n"], 2)
        self.assertAlmostEqual(out["accuracy"], 0.5)
        self.assertAlmostEqual(out["unparsed_rate"], 0.5)

    def test_breaks_down_by_level(self):
        qs = [q("001-1", "A", level="1"), q("001-2", "B", level="3")]
        preds = {"001-1": "A", "001-2": "X"}
        out = summarise(qs, preds)
        self.assertAlmostEqual(out["by_level"]["1"], 1.0)
        self.assertAlmostEqual(out["by_level"]["3"], 0.0)

    def test_breaks_down_by_second_head(self):
        qs = [
            q("001-1", "A", head="Frame-Only"),
            q("001-2", "B", head="Frames & Audio"),
        ]
        preds = {"001-1": "A", "001-2": "X"}
        out = summarise(qs, preds)
        self.assertAlmostEqual(out["by_second_head"]["Frame-Only"], 1.0)
        self.assertAlmostEqual(out["by_second_head"]["Frames & Audio"], 0.0)

    def test_group_score_is_reported(self):
        qs = [q(f"001-{i}", "A", group_type="logic") for i in range(1, 5)]
        preds = {"001-1": "X", "001-2": "A", "001-3": "A", "001-4": "A"}
        out = summarise(qs, preds)
        self.assertAlmostEqual(out["group_score"], 0.0)
        self.assertAlmostEqual(out["accuracy"], 0.75)

    def test_empty_input_does_not_divide_by_zero(self):
        out = summarise([], {})
        self.assertEqual(out["n"], 0)
        self.assertEqual(out["accuracy"], 0.0)
        self.assertEqual(out["unparsed_rate"], 0.0)


class TestConfigs(unittest.TestCase):
    def test_six_named_configs_exist(self):
        for name in (
            "baseline-uniform",
            "visual-only",
            "+asr-gt",
            "+asr-whisper",
            "+ocr",
            "full",
        ):
            self.assertIn(name, CONFIGS)

    def test_baseline_does_no_retrieval(self):
        cfg = get_config("baseline-uniform")
        self.assertFalse(cfg["retrieval"])

    def test_visual_only_zeroes_text_sources(self):
        w = get_config("visual-only")["weights"]
        self.assertEqual(w["asr"], 0.0)
        self.assertEqual(w["ocr"], 0.0)
        self.assertGreater(w["siglip"], 0.0)

    def test_asr_configs_differ_only_in_source(self):
        gt, wh = get_config("+asr-gt"), get_config("+asr-whisper")
        self.assertEqual(gt["weights"], wh["weights"])
        self.assertEqual(gt["asr_source"], "subtitle")
        self.assertEqual(wh["asr_source"], "whisper")

    def test_ocr_weight_only_nonzero_where_intended(self):
        self.assertEqual(get_config("+asr-gt")["weights"]["ocr"], 0.0)
        self.assertGreater(get_config("+ocr")["weights"]["ocr"], 0.0)

    def test_unknown_config_raises(self):
        with self.assertRaises(KeyError):
            get_config("does-not-exist")


if __name__ == "__main__":
    unittest.main()
