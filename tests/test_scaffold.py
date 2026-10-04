"""Guards the port: these are the exact breakages found when reading AIC source."""
import unittest


class TestImports(unittest.TestCase):
    def test_core_modules_import(self):
        from core.services.vector_store_optimized import QdrantService
        from core.services.elasticsearch_service import ElasticsearchService
        from online.backend.fusion import rrf_weighted_fuse
        self.assertTrue(callable(rrf_weighted_fuse))
        self.assertTrue(QdrantService and ElasticsearchService)


class TestVideoNameRegex(unittest.TestCase):
    """AIC's regex only accepted L01_V001; Video-MME-v2 names are '001'..'800'."""

    def test_videomme_name_is_accepted(self):
        from core.utils.parsing import is_known_video_name
        self.assertTrue(is_known_video_name("001"))
        self.assertTrue(is_known_video_name("800"))

    def test_aic_name_still_accepted(self):
        from core.utils.parsing import is_known_video_name
        self.assertTrue(is_known_video_name("L01_V001"))

    def test_junk_rejected(self):
        from core.utils.parsing import is_known_video_name
        self.assertFalse(is_known_video_name(""))
        self.assertFalse(is_known_video_name("not-a-video"))

    def test_metadata_not_flagged_for_videomme_payload(self):
        """With a '001' video name and a frame index present, nothing needs repair."""
        from core.utils.parsing import metadata_needs_repair
        payload = {
            "video": {"name": "001"},
            "frame": {"index": 120, "timestamp_seconds": 4.0},
            "file_path": "keyframes/001/0003_1.webp",
        }
        self.assertFalse(metadata_needs_repair(payload))


class TestFusionSurface(unittest.TestCase):
    def test_enrich_results_metadata_is_gone(self):
        """It only hydrated the caption collection, which this project drops."""
        import online.backend.fusion as fusion
        self.assertFalse(hasattr(fusion, "enrich_results_metadata"))

    def test_rrf_signature_is_unchanged(self):
        """Later tasks call this with exactly these kwargs."""
        import inspect
        from online.backend.fusion import rrf_weighted_fuse
        params = inspect.signature(rrf_weighted_fuse).parameters
        self.assertEqual(
            list(params), ["results_by_model", "k", "weights", "topn"]
        )


if __name__ == "__main__":
    unittest.main()
