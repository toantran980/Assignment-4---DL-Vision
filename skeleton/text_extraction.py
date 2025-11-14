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
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

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
    # Convert to grayscale
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Optionally resize to target_w preserving aspect ratio
    height, width = img_gray.shape
    if width > target_w:
        scale = target_w / width
        new_width = target_w
        new_height = int(height * scale)
        img_gray = cv2.resize(img_gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Apply blur and Otsu thresholding
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Small morphological closing to join broken characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_closed = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, kernel)

    # Return processed grayscale/thresholded numpy array
    return img_closed

    #raise NotImplementedError("preprocess: implement grayscale resize/blur/threshold/morphology")


# ---------- OCR wrapper (defined and used consistently) ----------
def run_tesseract_image_to_data(img_gray, psm=6, oem=3):
    """Return pytesseract.image_to_data dict for the provided grayscale image (numpy array)."""
    # TODO: implement wrapper that converts numpy array to PIL.Image and calls
    #       pytesseract.image_to_data(..., output_type=pytesseract.Output.DICT, config=cfg)
    #       Where cfg should include `--oem {oem} --psm {psm}`

    pil_image = Image.fromarray(img_gray)
    cfg = f'--oem {oem} --psm {psm}'
    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT, config=cfg)
    return data

    #raise NotImplementedError("run_tesseract_image_to_data: call pytesseract.image_to_data here")

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
    if not tok or not isinstance(tok, str):
        return None
    
    # Strip currency symbols and whitespace
    tok = tok.replace('$', '').strip()
    
    if not tok:
        return None

    # Normalize common OCR confusions (O->0, o->0)
    ocr_fixes = {
        'O': '0', 'o': '0', 'Q': '0',
        'l': '1', 'I': '1', '|': '1',
        'S': '5', 's': '5',
        'B': '8', 'b': '8',
        'Z': '2', 'z': '2',
    }
    
    for old_char, new_char in ocr_fixes.items():
        tok = tok.replace(old_char, new_char)

    # Convert commas to dots when appropriate
    comma_count = tok.count(',')
    dot_count = tok.count('.')
    
    if comma_count == 1 and dot_count == 0:
        tok = tok.replace(',', '.')
    elif comma_count > 0 and dot_count > 0:
        last_comma_pos = tok.rfind(',')
        last_dot_pos = tok.rfind('.')
        
        if last_comma_pos > last_dot_pos:
            tok = tok.replace('.', '').replace(',', '.')
        else:
            tok = tok.replace(',', '')
    elif comma_count > 1:
        tok = tok.replace(',', '')

    # Remove non-numeric characters
    tok = re.sub(r'[^\d.]', '', tok)
    
    if not tok or tok == '.':
        return None
    
    if tok.count('.') > 1:
        parts = tok.split('.')
        tok = ''.join(parts[:-1]) + '.' + parts[-1]
    
    if tok.startswith('.'):
        tok = '0' + tok
    if tok.endswith('.'):
        tok = tok + '0'
    
    # Parse float, round to 2 decimals
    try:
        price = float(tok)
        
        if price < 0 or price > 99999:
            return None
        
        return round(price, 2)
        
    except (ValueError, AttributeError, OverflowError):
        return None
    
    #raise NotImplementedError("normalize_price_token: TODO implement token -> float conversion")

def token_looks_like_price(token):
    """
    Heuristic to quickly determine whether a text token could represent a price.
    """
    # TODO: implement quick heuristics similar to original: presence of $ . , or short digit groups
    return PRICE_RE.match(token) or ANY_NUM_RE.match(token)

    #raise NotImplementedError("token_looks_like_price: TODO implement heuristic")

def is_footer_line(line_text):
    """
    Return True if line_text looks like a footer (survey, website, phone, etc).
    """
    # TODO: implement detection using SURVEY_FOOTERS and PHONE_RE
    return any(footer in line_text.lower() for footer in SURVEY_FOOTERS) or PHONE_RE.search(line_text)

    #raise NotImplementedError("is_footer_line: TODO implement footer detection")

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
    # Ignore generic words
    generic_words = ['receipt', 'invoice', 'store', 'sales', 'cashier', 
                     'date', 'time', 'order', 'number', 'phone', 'address']
    
    # Filter out empty and generic lines
    candidates = []
    for line in top_lines[:10]:  # Look at more lines
        line = line.strip()
        if len(line) < 2:
            continue
        line_lower = line.lower()
        if any(generic in line_lower for generic in generic_words):
            continue
        if line_lower.replace('-', '').replace('*', '').replace(' ', '').isdigit():
            continue
        candidates.append(line)
    
    if not candidates:
        return 'Unknown Store'
    
    # Check for known stores (including partial matches)
    for line in candidates:
        for store in COMMON_STORES:
            if store in line.lower():
                # Return the full line, not just the matched part
                return line.strip()
    
    # Return first reasonable candidate
    return candidates[0]

    #raise NotImplementedError("pick_store_name: TODO implement store name selection")

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

    # Build list of token dicts with numeric left/top/height/conf
    tokens = []
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > 0 and data['text'][i].strip():
            token = {
                'text': data['text'][i].strip(),
                'left': int(data['left'][i]),
                'top': int(data['top'][i]),
                'height': int(data['height'][i]),
                'conf': int(data['conf'][i])
            }
            tokens.append(token)
    
    if not tokens:
        return []
    
    # Sort by top, then left
    tokens.sort(key=lambda t: (t['top'], t['left']))
    
    # Calculate median height for dynamic tolerance
    heights = [t['height'] for t in tokens if t['height'] > 0]
    median_height = sorted(heights)[len(heights)//2] if heights else 20
    line_tolerance = median_height * 0.5
    
    # Group into lines
    lines = []
    current_line = [tokens[0]]
    
    for token in tokens[1:]:
        # Check if token belongs to current line (vertical overlap)
        prev = current_line[-1]
        if abs(token['top'] - prev['top']) <= line_tolerance:
            current_line.append(token)
        else:
            # Save current line and start new one
            current_line.sort(key=lambda t: t['left'])
            line_text = ' '.join(t['text'] for t in current_line if t['text'])
            lines.append({'tokens': current_line, 'line_text': line_text})
            current_line = [token]
    
    if current_line:
        current_line.sort(key=lambda t: t['left'])
        line_text = ' '.join(t['text'] for t in current_line if t['text'])
        lines.append({'tokens': current_line, 'line_text': line_text})
    
    return lines

    #raise NotImplementedError("cluster_tokens_into_lines: TODO implement clustering logic")

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
    line_text = line_obj['line_text']
    tokens = line_obj['tokens']
    
    # Skip if line is too short or looks like footer
    if len(line_text.strip()) < 2 or is_footer_line(line_text):
        return None, None, False
    
    # Mark is_total if line contains TOTAL_KEYWORDS
    is_total = any(keyword in line_text.lower() for keyword in TOTAL_KEYWORDS)
    
    # Detect qty @ unit patterns (e.g., '3 @ 0.29')
    qty_match = re.search(r'(\d+)\s*@\s*([\d.,]+)', line_text)
    if qty_match:
        qty = int(qty_match.group(1))
        unit_price = normalize_price_token(qty_match.group(2))
        if unit_price and unit_price > 0:
            total_price = round(qty * unit_price, 2)
            if total_price <= max_item_price or is_total:
                item_parts = line_text[:qty_match.start()].strip()
                return item_parts if item_parts else None, total_price, is_total
    
    # Collect numeric candidates in the line, prefer rightmost candidate with sufficient conf
    candidates = []
    for token in tokens:
        if token['conf'] >= conf_threshold and token_looks_like_price(token['text']):
            # Filter out phone-like or long-int tokens
            if PHONE_RE.search(token['text']) or LONG_INT_ONLY_RE.match(token['text']):
                continue
            candidates.append(token)
    
    if not candidates:
        return None, None, is_total
    
    # Prefer rightmost candidate with sufficient conf
    chosen_token = candidates[-1]
    
    # Normalize chosen candidate to a float price
    price = normalize_price_token(chosen_token['text'])
    
    # Validate price
    if price is None or price <= 0:
        return None, None, is_total
    
    # For non-total lines, enforce max price
    if not is_total and price > max_item_price:
        return None, None, is_total
    
    # Determine item name as text left of chosen token, cleaned up
    item_tokens = [t['text'] for t in tokens if t['left'] < chosen_token['left'] - 10]
    item_name = ' '.join(item_tokens).strip(' .:-_*')
    
    # Require reasonable item name for non-total lines
    if not is_total:
        if len(item_name) < 2:
            return None, None, is_total
        # Filter out single numbers or obvious garbage
        if item_name.replace('.', '').replace(',', '').replace(' ', '').isdigit():
            return None, None, is_total
    
    return item_name if item_name else None, price, is_total

    # NotImplementedError("parse_line_for_item: TODO implement item/price extraction heuristics")

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
    # read image with cv2.imread
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {path}")
    
    # preprocess
    img_preprocessed = preprocess(img_bgr, target_w=args.target_width)
    
    # run tesseract
    ocr_data = run_tesseract_image_to_data(img_preprocessed, psm=args.psm)
    
    # cluster tokens into lines
    clustered_lines = cluster_tokens_into_lines(ocr_data)
    if not clustered_lines:
        raise ValueError(f"No text lines found in image: {path}")
    
    # pick store name
    top_lines_text = [line['line_text'] for line in clustered_lines[:5]]
    store_name = pick_store_name(top_lines_text)

    # parse each line for items
    items_list = []
    computed_total = 0.0
    printed_total = None
    for line_obj in clustered_lines:
        item_name, price, is_total = parse_line_for_item(line_obj, args.conf, args.max_item)
        if item_name is not None:
            items_list.append((item_name, price))
            computed_total += price
        if is_total and price is not None:
            printed_total = price
    return store_name, items_list, round(computed_total, 2), printed_total

    #raise NotImplementedError("process_image_file: TODO implement full file processing pipeline")

# ---------- CSV I/O ----------
def write_csv_rows(rows, out_path):
    """
    Write rows (list of dict with keys 'store','item','amount') to CSV file with header.
    """

    # --- Final hard filter for footer / survey / $0 lines ---
    import re

    def _norm(s):
        return re.sub(r'[^a-z0-9]+', '', s.lower()) if isinstance(s, str) else ''

    def _is_survey_footer(store, item):
        ns, ni = _norm(store), _norm(item)
        bad = [
            'survey', 'giveus', 'giveun', 'walmartcom', 'walmartcoma', 'www',
            'http', 'netwk', 'arkyout', 'sui0in', 'arky0ut'
        ]
        return any(t in ns or t in ni for t in bad)

    before = len(rows)
    rows = [
        r for r in rows
        if not _is_survey_footer(r.get('store',''), r.get('item',''))
        and r.get('amount','').strip() not in ('$0', '$0.00', '0', '0.00')
    ]
    after = len(rows)
    print(f"[CLEAN] Removed {before - after} footer/survey/zero rows. Kept {after}.")

    # --- Write the filtered rows ---
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
