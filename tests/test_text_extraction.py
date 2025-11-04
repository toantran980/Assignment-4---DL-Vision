# test_text_extraction.py
import os
import csv
import time
import shutil
import sys
import tempfile
import unittest
import contextlib
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

# Import the student's module under test
import text_extraction as te


class TestTextExtraction(unittest.TestCase):
    def setUp(self):
        # Create a small white image to act as a fake receipt image for parsing tests
        self.tmp_image_file = Path("tmp_receipt_test.jpg")
        arr = np.full((120, 200, 3), 255, dtype=np.uint8)
        Image.fromarray(arr).save(self.tmp_image_file)

    def tearDown(self):
        try:
            self.tmp_image_file.unlink()
        except Exception:
            pass

    def test_tesseract_is_used(self):
        """
        Ensure the OCR wrapper calls pytesseract.image_to_data when run_tesseract_image_to_data is invoked.
        Uses mocking so no real OCR runs.
        """
        # tiny grayscale image array
        img = np.zeros((8, 8), dtype=np.uint8)

        called = {'ok': False}

        def fake_image_to_data(pil_img, output_type=None, config=None):
            called['ok'] = True
            # minimal structure similar to pytesseract Output.DICT
            return {'level': [], 'page_num': [], 'block_num': [], 'par_num': [], 'line_num': [],
                    'word_num': [], 'left': [], 'top': [], 'width': [], 'height': [], 'conf': [], 'text': []}

        with mock.patch.object(te.pytesseract, 'image_to_data', side_effect=fake_image_to_data) as patched:
            out = te.run_tesseract_image_to_data(img, psm=6, oem=3)
            patched.assert_called()
            self.assertTrue(called['ok'], "pytesseract.image_to_data should be called by run_tesseract_image_to_data()")
            self.assertIsInstance(out, dict)
            self.assertIn('text', out)

    def test_write_csv_columns_and_sample_rows(self):
        """
        Test that write_csv_rows writes CSV with header store,item,amount and that sample rows are present.
        """
        rows = [
            {'store': "WAL*MART t", 'item': "OPEN 24 HOURS", 'amount': "$24.00"},
            {'store': "TRADER JOE'S", 'item': "TOMATOES CRUSHED NO SALT", 'amount': "$1.59"},
            {'store': "TRADER JOE'S", 'item': "ORGANIC OATMEAL", 'amount': "$2.69"},
        ]
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tf.close()
        out_path = tf.name
        try:
            te.write_csv_rows(rows, out_path)
            with open(out_path, newline='', encoding='utf-8') as fh:
                reader = csv.reader(fh)
                header = next(reader)
                self.assertEqual(header, ['store', 'item', 'amount'], "CSV header must be ['store','item','amount']")
                content = list(reader)
                # Check rows exist
                self.assertIn([rows[0]['store'], rows[0]['item'], rows[0]['amount']], content)
                self.assertIn([rows[1]['store'], rows[1]['item'], rows[1]['amount']], content)
        finally:
            try:
                os.remove(out_path)
            except Exception:
                pass

    def test_process_image_file_parses_sample_receipt_lines(self):
        """
        Patch run_tesseract_image_to_data to return synthetic tokens to simulate realistic OCR output,
        and assert process_image_file extracts store and several expected items/prices.
        """
        # Build synthetic tesseract response arrays (tokenized)
        texts = [
            "TRADER JOE'S",
            "TOMATOES", "CRUSHED", "NO", "SALT", "1.59",
            "ORGANIC", "OATMEAL", "2.69",
            "MINI-PEARL", "TOMATOES", "2.49",
        ]
        lefts = [10, 10, 70, 120, 160, 320, 10, 80, 320, 10, 120, 320]
        tops =  [10, 40, 40, 40, 40, 40, 70, 70, 70, 100, 100, 100]
        heights = [10] * len(texts)
        confs = ['96', '85', '80', '80', '82', '95', '88', '87', '95', '86', '82', '95']

        fake_data = {
            'level': [5]*len(texts),
            'page_num': [1]*len(texts),
            'block_num': [1]*len(texts),
            'par_num': [1]*len(texts),
            'line_num': [1]*len(texts),
            'word_num': list(range(1, len(texts)+1)),
            'left': lefts,
            'top': tops,
            'width': [50]*len(texts),
            'height': heights,
            'conf': confs,
            'text': texts
        }

        with mock.patch.object(te, 'run_tesseract_image_to_data', return_value=fake_data) as patched_ocr:
            class Args: pass
            args = Args()
            args.psm = 6
            args.conf = 50.0
            args.max_item = 200.0
            args.target_width = 1200

            store, items, computed_total, printed_total = te.process_image_file(str(self.tmp_image_file), args)
            patched_ocr.assert_called()
            self.assertIsInstance(store, str)
            self.assertTrue('trader' in store.lower() or 'trader joe' in store.lower(),
                            f"Expected store resembling Trader Joe's, got '{store}'")

            # check tomatoes ~1.59
            found_tomatoes = any('tomato' in name.lower() and abs(price - 1.59) < 0.03 for name, price in items)
            self.assertTrue(found_tomatoes, f"Expected tomatoes priced ~1.59 in items: {items}")

            # oatmeal ~2.69
            found_oatmeal = any('oatmeal' in name.lower() and abs(price - 2.69) < 0.03 for name, price in items)
            self.assertTrue(found_oatmeal, f"Expected oatmeal priced ~2.69 in items: {items}")

            # mini-pearl tomatoes ~2.49
            found_mini = any(('mini' in name.lower() and 'tomato' in name.lower() and abs(price - 2.49) < 0.03)
                             for name, price in items)
            self.assertTrue(found_mini, f"Expected mini-pearl tomatoes priced ~2.49 in items: {items}")

            # computed_total equals sum(items)
            expected_sum = round(sum(p for (_, p) in items), 2)
            self.assertAlmostEqual(computed_total, expected_sum, places=2,
                                   msg=f"computed_total {computed_total} != sum of items {expected_sum}")

    def test_is_footer_line_and_pick_store_name_helpers(self):
        """Small additional checks for helper functions to increase robustness of grading."""
        # Footer lines should be detected
        self.assertTrue(te.is_footer_line("Visit survey at www.survey.com"), "Footer detection failed for survey line")
        self.assertTrue(te.is_footer_line("Call 1-800-123-4567"), "Footer detection failed for phone-like line")
        # pick_store_name: should pick a common store from top lines
        sample_top = ["WAL*MART t", "OPEN 24 HOURS", "Some Address"]
        store = te.pick_store_name(sample_top)
        self.assertTrue(isinstance(store, str) and len(store) > 0)

    def test_integration_with_real_tesseract(self):
        """
        Integration test that runs the student's actual text_extraction.main() using a real Tesseract.
        This test overrides text_extraction.pytesseract.pytesseract.tesseract_cmd,
        so students' hardcoded paths won't block the test.
        """

        # Discover tesseract: prefer PATH, else try the common Windows path you mentioned
        tpath = shutil.which("tesseract")
        if not tpath:
            candidate = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if Path(candidate).exists():
                tpath = candidate

        if not tpath:
            self.skipTest("Tesseract binary not found on PATH or common Windows location; skipping integration test.")

        # Save original and override the student's module tesseract command
        orig_cmd = None
        try:
            orig_cmd = getattr(te.pytesseract.pytesseract, 'tesseract_cmd', None)
        except Exception:
            orig_cmd = None
        te.pytesseract.pytesseract.tesseract_cmd = str(tpath)

        # Determine receipts folder and output CSV (configurable via env)
        receipts_folder = Path(os.environ.get("TEST_RECEIPT_FOLDER", "../datasets/receipts"))
        out_csv = Path(os.environ.get("TEST_OUT_CSV", "shopping_summary_integration.csv"))

        if not receipts_folder.exists():
            # restore and skip
            if orig_cmd is not None:
                te.pytesseract.pytesseract.tesseract_cmd = orig_cmd
            self.skipTest(f"Receipts folder {receipts_folder} not found; cannot run integration test.")

        # Run the student's main() with arguments by temporarily replacing sys.argv
        saved_argv = sys.argv[:]
        try:
            sys.argv = ["text_extraction.py", "--folder", str(receipts_folder), "--out", str(out_csv),
                        "--psm", "6", "--conf", "55", "--max-item", "200", "--target-width", "1200"]
            # Run main (this will perform OCR using the real Tesseract)
            te.main()
        finally:
            sys.argv = saved_argv
            # restore tesseract cmd
            if orig_cmd is not None:
                te.pytesseract.pytesseract.tesseract_cmd = orig_cmd

        # Wait briefly for CSV to appear
        timeout = 30.0
        elapsed = 0.0
        while not out_csv.exists() and elapsed < timeout:
            time.sleep(0.2)
            elapsed += 0.2

        self.assertTrue(out_csv.exists(), msg=f"Integration output CSV {out_csv} not created by text_extraction.main()")

        # Read CSV and assert header & presence of a few expected lines
        found_rows = []
        with open(out_csv, newline='', encoding='utf-8') as fh:
            reader = csv.reader(fh)
            header = next(reader)
            self.assertEqual(header, ['store', 'item', 'amount'], "Integration CSV header mismatch")
            for row in reader:
                found_rows.append((row[0].strip().lower(), row[1].strip().lower(), row[2].strip()))

        # Basic content checks: ensure at least one Trader and one Walmart-like store row exists
        has_trader = any('trader' in s for (s, _, _) in found_rows)
        has_walmart = any('wal' in s or 'walmart' in s for (s, _, _) in found_rows)
        self.assertTrue(has_trader or has_walmart, "Integration run did not yield Trader/Walmart entries")

        # Check that at least one expected item keyword appears
        expected_item_keywords = ['tomato', 'oatmeal', 'total']
        found_keyword_matches = 0
        for (_, item, _) in found_rows:
            for kw in expected_item_keywords:
                if kw in item:
                    found_keyword_matches += 1
                    break
        self.assertGreaterEqual(found_keyword_matches, 1, "Integration run did not capture expected item keywords (e.g., 'tomato', 'oatmeal')")

        # cleanup output CSV (best-effort)
        try:
            out_csv.unlink()
        except Exception:
            pass


if __name__ == '__main__':
    unittest.main()
