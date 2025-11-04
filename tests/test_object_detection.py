# test_object_detection.py
import io
import unittest
import contextlib
from types import SimpleNamespace
from unittest import mock
from pathlib import Path
import numpy as np
import pandas as pd

# Import student's module
import object_detection as od


class FakeResults:
    """A tiny fake results object that mimics the interface used in the student's code:
       results.pandas().xyxy[0] -> pandas.DataFrame
    """
    def __init__(self, df):
        self._df = df

    def pandas(self):
        return SimpleNamespace(xyxy=[self._df])


class DummyModel:
    """A dummy model that returns the FakeResults when called."""
    def __init__(self, results):
        self._results = results

    def __call__(self, image):
        return self._results


class TestObjectDetection(unittest.TestCase):
    def setUp(self):
        # Build a deterministic, small DataFrame with exactly cat, dog, person rows
        # xmin, ymin, xmax, ymax, confidence, name
        rows = [
            {'xmin': 10.0, 'ymin': 20.0, 'xmax': 110.0, 'ymax': 220.0, 'confidence': 0.95, 'name': 'cat'},
            {'xmin': 130.0, 'ymin': 40.0, 'xmax': 330.0, 'ymax': 340.0, 'confidence': 0.92, 'name': 'dog'},
            {'xmin': 50.0, 'ymin': 10.0, 'xmax': 200.0, 'ymax': 400.0, 'confidence': 0.99, 'name': 'person'},
        ]
        self.df = pd.DataFrame(rows)

        # Fake results and dummy model that returns them
        self.fake_results = FakeResults(self.df)
        self.dummy_model = DummyModel(self.fake_results)

        # Create a tiny dummy image (PIL-compatible). We won't rely on image content.
        from PIL import Image
        self.tmp_image = Image.fromarray(np.uint8(np.zeros((64, 64, 3))))

        # We'll patch od.load_image to return this image for integration test
        self.tmp_image_path = Path("tmp_test_image.jpg")
        self.tmp_image.save(self.tmp_image_path)

    def tearDown(self):
        # Clean tmp image if created
        try:
            self.tmp_image_path.unlink()
        except Exception:
            pass

    def test_load_model_calls_torch_hub(self):
        """Ensure load_model calls torch.hub.load (mocked) and returns its result."""
        with mock.patch('object_detection.torch.hub.load') as mocked_hub_load:
            mocked_hub_load.return_value = "SOME_MODEL_OBJECT"
            model = od.load_model(model_name='yolov5s', force_reload=False)
            mocked_hub_load.assert_called()  # ensure it was called at least once
            self.assertEqual(model, "SOME_MODEL_OBJECT")

    def test_extract_predictions_prints_objects_and_confidences(self):
        """Test extract_predictions prints names and confidences for each detected object."""
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            df_out = od.extract_predictions(self.fake_results)
        output = f.getvalue()
        # Should return the same dataframe
        pd.testing.assert_frame_equal(df_out.reset_index(drop=True), self.df.reset_index(drop=True))
        # Check printed names and confidences appear
        self.assertIn("Objects detected", output)
        for name, conf in zip(self.df['name'], self.df['confidence']):
            self.assertIn(name, output)
            # confidence formatted with two decimals
            self.assertIn(f"{conf:.2f}", output)

    def test_print_bounding_boxes_function_and_format(self):
        """
        The student's code must provide a function that prints bounding box coordinates.
        We expect a function named `print_bounding_boxes(predictions)` that prints one line per detection
        including label and 4 coords. Example line: 'cat: 10,20,110,220'
        """
        self.assertTrue(hasattr(od, 'print_bounding_boxes'),
                        msg="Missing function `print_bounding_boxes(predictions)`. Please add it to object_detection.py")

        # Call the function and capture stdout
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            # The student's function should accept the predictions DataFrame (same as extract_predictions produces)
            od.print_bounding_boxes(self.df)

        out = f.getvalue().strip().splitlines()
        # Expect 3 lines (one per detection)
        self.assertGreaterEqual(len(out), 3, msg="print_bounding_boxes should print a line per detection.")
        # For each row in df, ensure a line contains the label and the 4 coords
        for row in self.df.itertuples(index=False):
            label = row.name
            coords = (int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax))
            # Look for a line containing the label and all coords (as integers, separated by commas or spaces)
            found = False
            for line in out:
                if label in line:
                    # check all coords present as integers in line
                    if all(str(c) in line for c in coords):
                        found = True
                        break
            self.assertTrue(found, msg=f"Bounding box line for '{label}' with coords {coords} not found in output:\n" + "\n".join(out))

    def test_run_object_detection_prints_only_cat_dog_person(self):
        """
        Integration-like test: patch load_model to return a dummy model that yields the fake results
        (cat,dog,person). Patch display_with_opencv to avoid GUI. Capture stdout and assert
        the printed detected object names are exactly the set {'cat','dog','person'} and no others.
        """
        # Patch load_model so run_object_detection uses our dummy_model
        with mock.patch('object_detection.load_model', return_value=self.dummy_model):
            # Patch load_image to return our small image (so run_object_detection can call it)
            with mock.patch('object_detection.load_image', return_value=self.tmp_image):
                # Patch display to avoid any GUI calls
                with mock.patch('object_detection.display_with_opencv') as patched_display:
                    f = io.StringIO()
                    with contextlib.redirect_stdout(f):
                        # call main pipeline with our tmp image path and model name (ignored by patched loader)
                        od.run_object_detection(img_path=str(self.tmp_image_path), model_name='yolov5s')
                    output = f.getvalue()

                    # Extract any printed object lines (extract_predictions prints names)
                    # Ensure the three required names appear and no other names appear
                    required = {'cat', 'dog', 'person'}
                    found = set()
                    for name in required:
                        self.assertIn(name, output, msg=f"Expected object '{name}' not printed by run_object_detection output.")
                        found.add(name)

                    # Gather all names printed by extract_predictions (we look for lines starting with ' - ')
                    printed_names = set()
                    for line in output.splitlines():
                        if line.strip().startswith('- '):
                            # line looks like " - name: 0.95"
                            try:
                                rest = line.strip()[2:]
                                name_part = rest.split(':')[0].strip()
                                printed_names.add(name_part)
                            except Exception:
                                pass
                    # printed_names should be exactly the required set
                    self.assertEqual(printed_names, required,
                                     msg=f"run_object_detection printed unexpected object names: {printed_names}. Expected exactly {required}.")

                    # Also assert display_with_opencv was called (though patched)
                    patched_display.assert_called()

if __name__ == "__main__":
    unittest.main()
