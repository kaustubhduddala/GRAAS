import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from program.measure import trained, training  # noqa: E402


class RecommendTests(unittest.TestCase):
    def test_small_memory_gets_a_smaller_image_size(self):
        small, large = training.recommend("mps", 8), training.recommend("mps", 64)
        self.assertLess(small["imgsz"], large["imgsz"])
        self.assertLess(small["batch"], large["batch"])
        self.assertGreaterEqual(small["batch"], 1)

    def test_batch_is_a_power_of_two_and_fits(self):
        for device, memory in (("mps", 24), ("cuda", 12), ("cpu", 16), ("mps", None)):
            settings = training.recommend(device, memory)
            self.assertIn(settings["batch"], (1, 2, 4, 8, 16))
            self.assertIn(settings["imgsz"], training.IMGSZ_STEPS + (training.MIN_IMGSZ,))

    def test_fewer_sheets_get_more_epochs(self):
        few, many = training.recommend("mps", 24, 20), training.recommend("mps", 24, 400)
        self.assertGreater(few["epochs"], many["epochs"])
        self.assertTrue(50 <= many["epochs"] <= few["epochs"] <= 300)


class SavedSettingsTests(unittest.TestCase):
    def test_missing_or_bad_values_fall_back_to_defaults(self):
        with mock.patch.object(training, "load_settings", return_value={"training": {"epochs": "x", "imgsz": 960,
                                                                                    "device": "tpu"}}):
            saved = training.saved_settings()
        self.assertEqual((saved["epochs"], saved["imgsz"], saved["batch"], saved["device"]),
                         (training.DEFAULTS["epochs"], 960, None, "auto"))

    def test_out_of_memory_message_is_recognised(self):
        self.assertTrue(training.is_out_of_memory(RuntimeError("MPS ... kIOGPUCommandBufferCallbackErrorOutOfMemory")))
        self.assertFalse(training.is_out_of_memory(ValueError("bad label")))


class LandmarkLayout(unittest.TestCase):
    """The pose model's labels: a base and a tip keep their names in a mirror image, a leaf's margins swap."""

    @staticmethod
    def _values(obj, size=(1000, 1000), **kwargs):
        return [float(v) for v in training.pose_line(obj, size, **kwargs).split()]

    def _slot(self, values, slot):
        return values[5 + 3 * slot: 8 + 3 * slot]

    def test_each_class_uses_its_own_pair(self):
        panicle = {"cls": "panicle", "box": [100, 100, 300, 800], "keypoints": {"base": [200, 800], "tip": [210, 100]}}
        leaf = {"cls": "leaf", "box": [400, 400, 500, 500], "keypoints": {"a": [480, 450], "b": [420, 452]}}
        p, l = self._values(panicle), self._values(leaf)
        self.assertEqual(len(p), 5 + 3 * trained.POSE_KEYPOINTS)
        self.assertEqual(self._slot(p, 0), [0.2, 0.8, 2.0])                  # base
        self.assertEqual(self._slot(p, 2)[2], 0.0)                           # the leaf pair is unused
        self.assertEqual(self._slot(l, 0)[2], 0.0)
        self.assertEqual(self._slot(l, 2), [0.42, 0.452, 2.0])               # the left margin comes first

    def test_mirroring_keeps_the_leaf_order(self):
        """What the trainer does to a mirrored image (x -> 1 - x, keypoints reordered by FLIP_IDX) must give the
        label the same rule would write for the mirrored sheet."""
        leaf = {"cls": "leaf", "box": [100, 100, 300, 200], "keypoints": {"a": [120, 150], "b": [280, 160]}}
        values = self._values(leaf)
        points = [self._slot(values, k) for k in range(trained.POSE_KEYPOINTS)]
        mirrored = [[1 - x if v else 0.0, y, v] for x, y, v in points]
        flipped = [mirrored[k] for k in trained.FLIP_IDX]
        mirror_leaf = {"cls": "leaf", "box": [700, 100, 900, 200], "keypoints": {"a": [880, 150], "b": [720, 160]}}
        expected = self._values(mirror_leaf)
        for k in range(trained.POSE_KEYPOINTS):
            self.assertEqual([round(v, 6) for v in flipped[k]], self._slot(expected, k))

    def test_tile_offset_and_scale(self):
        seed = {"cls": "seed", "box": [1000, 1000, 1040, 1020], "keypoints": {"base": [1000, 1010], "tip": [1040, 1010]}}
        values = self._values(seed, size=(100, 100), offset=(480, 480), scale=0.5)
        self.assertEqual(self._slot(values, 0), [0.2, 0.25, 2.0])
        self.assertEqual(self._slot(values, 1), [0.4, 0.25, 2.0])

    def test_seed_scale(self):
        self.assertEqual(trained.seed_scale(None), 1.0)
        self.assertEqual(trained.seed_scale(160.0), 0.5)
        self.assertEqual(trained.seed_scale(55.0, None), 1.0)                # model trained on full-resolution tiles
        self.assertEqual(trained.seed_scale(10.0), 2.0)                      # never enlarged more than twice


if __name__ == "__main__":
    unittest.main()
