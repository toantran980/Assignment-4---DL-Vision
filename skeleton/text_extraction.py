#!/usr/bin/env python3
# text_extraction.py
"""
Receipt OCR extraction using Tesseract (pytesseract) with improved heuristics.

Usage:
    python text_extraction.py --zip receipts.zip --out shopping_summary.csv
    python text_extraction.py --folder ./receipts --conf 50 --max-item 150

Only uses pytesseract/Tesseract; tune --conf and --max-item for your data.
"""

import argparse
import os
import re
import csv
import zipfile
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import pytesseract

# Tesseract executable path override (if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------- constants & regex ----------
PRICE_RE = re.compile(r'\$?\s*\d{1,3}(?:[.,]\d{1,2})?$')
ANY_NUM_RE = re.compile(r'[\d\.,]+')
PHONE_RE = re.compile(r'(\+?\d[\d\-\s().]{6,}\d)')
LONG_INT_ONLY_RE = re.compile(r'^\d{6,}$')
SURVEY_FOOTERS = ['survey', 'visit', 'www', 'http', 'feedback', 'give us', 'thank you', 'receipt id', 'ref', 'approval', 'terminal', 'tc#', 'trans id', 'merchant', 'card', 'visa', 'mastercard', 'amex', 'transaction']
TOTAL_KEYWORDS = ['total', 'amount due', 'grand total', 'subtotal', 'balance', 'amount']
COMMON_STORES = ['walmart', 'trader', 'trader joe', 'whole foods', 'safeway', 'target', 'cvs', 'walgreens', 'aldi', 'sprouts']

# ---------- filesystem helpers ----------
def extract_zip_to_folder(zip_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(out_dir)
    return out_dir

def list_images(folder):
    exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
    files = [str(p) for p in Path(folder).rglob('*') if p.suffix.lower() in exts]
    return sorted(files)

# ---------- image preprocessing ----------
def preprocess(img_bgr, target_w=1200):
    """
    Preprocess BGR image (numpy array) into a thresholded grayscale image suitable for OCR.
    Steps to implement (students):
      - convert to grayscale
      - optionally resize to target_w preserving aspect ratio
      - apply blur and Otsu thresholding
      - small morphological closing to join broken characters
      - return processed grayscale/thresholded numpy array
    """
    # TODO: implement preprocessing pipeline using cv2 operations described above
    raise NotImplementedError("preprocess: implement grayscale resize/blur/threshold/morphology")

# ---------- OCR wrapper (defined and used consistently) ----------
def run_tesseract_image_to_data(img_gray, psm=6, oem=3):
    """Return pytesseract.image_to_data dict for the provided grayscale image (numpy array)."""
    # TODO: implement wrapper that converts numpy array to PIL.Image and calls
    #       pytesseract.image_to_data(..., output_type=pytesseract.Output.DICT, config=cfg)
    #       Where cfg should include `--oem {oem} --psm {psm}`
    raise NotImplementedError("run_tesseract_image_to_data: call pytesseract.image_to_data here")

# ---------- utilities ----------
def normalize_price_token(tok):
    """
    Normalize price-like token into float (rounded to 2 dp), or return None if invalid.
    Students should:
      - strip currency symbols and whitespace
      - normalize common OCR confusions (O->0, o->0)
      - convert commas to dots when appropriate
      - remove non-numeric characters, parse float, round to 2 decimals
    """
    # TODO: implement normalization and robust parsing
    raise NotImplementedError("normalize_price_token: TODO implement token -> float conversion")

def token_looks_like_price(token):
    """
    Heuristic to quickly determine whether a text token could represent a price.
    """
    # TODO: implement quick heuristics similar to original: presence of $ . , or short digit groups
    raise NotImplementedError("token_looks_like_price: TODO implement heuristic")

def is_footer_line(line_text):
    """
    Return True if line_text looks like a footer (survey, website, phone, etc).
    """
    # TODO: implement detection using SURVEY_FOOTERS and PHONE_RE
    raise NotImplementedError("is_footer_line: TODO implement footer detection")

def pick_store_name(top_lines):
    """
    Choose most likely store name from the top lines of the receipt.
    Heuristics:
      - ignore generic words like 'receipt'/'invoice' etc.
      - prefer known common stores found in COMMON_STORES
      - prefer lines with letters and not many digits
      - fallback to first non-empty line or 'Unknown Store'
    """
    # TODO: implement selection heuristics described above
    raise NotImplementedError("pick_store_name: TODO implement store name selection")

# ---------- token -> lines clustering ----------
def cluster_tokens_into_lines(data):
    """
    Cluster Tesseract token dict into spatially grouped text lines.
    Input: data dict as returned by pytesseract Output.DICT with keys like 'text','left','top','height','conf'
    Output: list of {'tokens': [...], 'line_text': '...'} sorted top->bottom
    """
    # TODO: implement token parsing and clustering:
    #   - build list of token dicts with numeric left/top/height/conf
    #   - sort tokens by top, left
    #   - group tokens whose 'top' is close into line buckets
    #   - compute line_text by joining token['text'] by left order
    raise NotImplementedError("cluster_tokens_into_lines: TODO implement clustering logic")

# ---------- line parsing ----------
def parse_line_for_item(line_obj, conf_threshold, max_item_price):
    """
    Parse a clustered line for an item / price / whether it's a total line.
    Returns: (item_name or None, price or None, is_total_flag)
    Students should implement heuristics:
      - detect qty @ unit patterns (e.g., '3 @ 0.29')
      - collect numeric candidates in the line, prefer rightmost candidate with sufficient conf
      - filter out phone-like or long-int tokens
      - normalize chosen candidate to a float price
      - determine item name as text left of chosen token, cleaned up
      - mark is_total if line contains TOTAL_KEYWORDS
    """
    # TODO: implement robust line parsing; this is the core logic used by process_image_file
    raise NotImplementedError("parse_line_for_item: TODO implement item/price extraction heuristics")

# ---------- process a single file ----------
def process_image_file(path, args):
    """
    Process a single image file path:
      - read image with cv2.imread
      - preprocess
      - run tesseract (run_tesseract_image_to_data)
      - cluster tokens into lines (cluster_tokens_into_lines)
      - pick store name (pick_store_name)
      - parse each line for items (parse_line_for_item)
      - compute computed_total as sum(items) and capture printed_total when found
    Returns: (store, items_list, computed_total, printed_total)
    """
    # TODO: implement orchestration using the helper functions above.
    raise NotImplementedError("process_image_file: TODO implement full file processing pipeline")

# ---------- CSV I/O ----------
def write_csv_rows(rows, out_path):
    """
    Write rows (list of dict with keys 'store','item','amount') to CSV file with header.
    """
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['store','item','amount'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

# ---------- CLI entry ----------
def main():
    parser = argparse.ArgumentParser(description='Receipt OCR extractor (Tesseract only) with robust heuristics.')
    parser.add_argument('--zip', type=str, default=None, help='zip file with images')
    parser.add_argument('--folder', type=str, default='../datasets/receipts', help='folder with images')
    parser.add_argument('--out', type=str, default='shopping_summary.csv', help='output CSV')
    parser.add_argument('--psm', type=int, default=6, help='tesseract PSM (page segmentation mode)')
    parser.add_argument('--conf', type=float, default=45.0, help='preferred min token confidence (0-100)')
    parser.add_argument('--max-item', type=float, default=200.0, help='max plausible per-item price (non-total)')
    parser.add_argument('--target-width', type=int, default=1200, help='preprocess resize width for OCR (higher = slower)')
    args = parser.parse_args()

    if not args.zip and not args.folder:
        raise SystemExit('Provide --zip or --folder containing receipt images.')

    img_folder = None
    if args.zip:
        img_folder = 'extracted_receipts'
        extract_zip_to_folder(args.zip, img_folder)
    else:
        img_folder = args.folder

    files = list_images(img_folder)
    if not files:
        raise SystemExit('No images found in folder.')

    rows = []
    for f in sorted(files):
        try:
            store, items, computed_total, printed_total = process_image_file(f, args)
        except Exception as e:
            print(f"[ERROR] processing {f}: {e}")
            continue
        for name, price in items:
            rows.append({'store': store, 'item': name, 'amount': f'${price:.2f}'})
        rows.append({'store': store, 'item': 'Total', 'amount': f'${computed_total:.2f}'})
        if printed_total is not None:
            rows.append({'store': store, 'item': 'Receipt_Total', 'amount': f'${printed_total:.2f}'})

    write_csv_rows(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")
    print('If you still see huge nonsense values: increase --conf (e.g. 55) and lower --max-item (e.g. 100).')

if __name__ == '__main__':
    main()
