# test_predict_my_digits.py
import os
import unittest
from pathlib import Path
import importlib

# Import the student's prediction module
import predict_my_digits as pm


class TestPredictMyDigits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # expected model filename per your requirement
        cls.model_name = Path("improved_digit_cnn.pth")
        # expected image files (must exist in working directory or provide correct relative path)
        cls.images = [
            Path("../datasets/digits/digit2.jpg"),
            Path("../datasets/digits/digit4.jpg"),
            Path("../datasets/digits/digit6.jpg"),
            Path("../datasets/digits/digit8.jpg"),
        ]
        # map expected ground-truth from filename -> int
        cls.expected = {
            "digit2.jpg": 2,
            "digit4.jpg": 4,
            "digit6.jpg": 6,
            "digit8.jpg": 8,
        }

    def test_model_file_exists(self):
        """Ensure the required model file improved_digit_cnn.pth is present in working dir."""
        self.assertTrue(self.model_name.exists(),
                        msg=f"Required model file '{self.model_name}' not found in working directory. Place the file there before running tests.")

    def test_image_files_exist(self):
        """Ensure all target digit images exist (skip test if not)."""
        # missing = [str(p) for p in self.images if not p.exists()]
        # if missing:
        #     self.skipTest(f"Missing image files required for this test: {missing}")
        # if present, pass
        for p in self.images:
            self.assertTrue(p.exists())

    def test_load_trained_model_returns_model(self):
        """Ensure load_trained_model can load the model file without raising (and returns a model)."""
        # if not self.model_name.exists():
        #     self.skipTest(f"Model file {self.model_name} not found; skipping load test.")
        # use CPU for deterministic behavior
        device = "cpu"
        try:
            model = pm.load_trained_model(str(self.model_name), device=__import__('torch').device(device))
        except Exception as e:
            self.fail(f"load_trained_model raised an exception when loading '{self.model_name}': {e}")
        self.assertIsNotNone(model, "load_trained_model returned None instead of a model instance.")

    def test_predict_each_digit_exact_match(self):
        """
        Run predictions on each provided digit image and assert the predicted number equals the
        true digit (extracted from filename). Each digit is a separate assert (subTest).
        Also verify the output string follows the required style:
          'Prediction for digitX.jpg: Y' (an optional ' (conf=...)' suffix is acceptable).
        """
        # if not self.model_name.exists():
        #     self.skipTest(f"Model file {self.model_name} not found; skipping prediction tests.")
        # missing = [str(p) for p in self.images if not p.exists()]
        # if missing:
        #     self.skipTest(f"Missing image files required for prediction test: {missing}")

        # Run predictions using the module function (device CPU)
        try:
            results = pm.predict_images(str(self.model_name),
                                        [str(p) for p in self.images],
                                        device_str="cpu",
                                        show=False)
        except Exception as e:
            self.fail(f"predict_images raised an exception: {e}")

        # Expect one result string per image in same order
        self.assertEqual(len(results), len(self.images), msg="predict_images returned unexpected number of results.")

        # Validate each output line: style and predicted integer
        for img_path, out in zip(self.images, results):
            with self.subTest(image=str(img_path)):
                basename = img_path.name
                # Required prefix
                expected_prefix = f"Prediction for {basename}: "
                self.assertTrue(out.startswith(expected_prefix),
                                msg=f"Output for {basename} does not start with required prefix '{expected_prefix}'. Actual: '{out}'")
                # Extract predicted token: everything after prefix, strip optional ' (conf=...)'
                rest = out[len(expected_prefix):].strip()
                # If conf suffix present, strip it
                if rest.endswith(")"):
                    # split on ' (' to remove the parenthetical conf if present
                    if " (" in rest:
                        rest = rest.split(" (", 1)[0].strip()
                    else:
                        # fallback: remove trailing ')' and anything inside parentheses
                        rest = rest.split(")", 1)[0].strip()
                # Now rest should be the predicted label string (like '2')
                try:
                    pred_int = int(rest)
                except Exception as e:
                    self.fail(f"Could not parse predicted integer from output for {basename}: '{out}' ({e})")

                # Assert exact equality to expected
                expected_int = self.expected[basename]
                self.assertEqual(pred_int, expected_int,
                                 msg=f"Prediction mismatch for {basename}: expected {expected_int}, got {pred_int}. Full output: '{out}'")


if __name__ == "__main__":
    unittest.main()
