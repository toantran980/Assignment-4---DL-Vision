#!/usr/bin/env python3
"""
skeleton_animal_classifier.py

Skeleton version of animal_classifier.py for assignments.
Keep function names/signatures intact. Replace core implementations with TODOs
so students implement the missing pieces.
"""

import os
import time
import argparse
import re
import json
from pathlib import Path
from collections import defaultdict

from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision

# fallback fuzzy matcher
import difflib
from urllib import request as urllib_request

# Small utility: try to import Levenshtein (optional)
try:
    import Levenshtein  # type: ignore
    _have_lev = True
except Exception:
    _have_lev = False


def normalize_text(s: str) -> str:
    """
    Normalize a text string for comparisons: lower-case, remove non-alphanumeric,
    collapse whitespace and trim.
    """
    s = s.lower()
    s = re.sub(r'[^0-9a-z]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def fuzzy_similarity(a: str, b: str) -> float:
    """
    Return a normalized similarity score (0..1) between two strings.
    Uses python-Levenshtein if available, otherwise fallback to difflib.
    """
    if _have_lev:
        return Levenshtein.ratio(a, b)
    else:
        return difflib.SequenceMatcher(None, a, b).ratio()


def list_image_files(folder: Path):
    """
    Return a sorted list of image files in a folder (by filename).
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()])


def simple_singular(word: str) -> str:
    """
    Very naive singularization helper for simple plural forms.
    """
    if word.endswith('ies'):
        return word[:-3] + 'y'
    if word.endswith('sses') or word.endswith('shes') or word.endswith('ches'):
        return word[:-2]
    if word.endswith('es') and len(word) > 3:
        return word[:-2]
    if word.endswith('s') and len(word) > 2:
        return word[:-1]
    return word


def load_imagenet_labels_from_torch(weights_enum):
    """
    Attempt to extract ImageNet label strings from a torchvision Weights enum object.
    TODO: students should inspect weights_enum.meta and return a list of 1000 label strings
    if available. Return None if labels cannot be found.
    """
    # TODO: implement extraction from weights_enum.meta["categories"] (or equivalent)
    raise NotImplementedError("load_imagenet_labels_from_torch not implemented")    


def download_imagenet_labels():
    """
    Download canonical ImageNet label list used in many PyTorch examples.
    TODO: implement network download and parsing of the 1000 labels.
    """
    # TODO: implement downloading from
    #  https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
    # and return list of 1000 label strings, or None on failure.
    raise NotImplementedError("download_imagenet_labels not implemented")


def load_imagenet_labels(model_name="resnet50"):
    """
    Attempt to obtain a list of ImageNet human-readable labels (1000 strings).
    Strategy (students should implement):
      1) try torchvision weights metadata
      2) fallback to downloading canonical list
      3) if both fail, return None
    Return the list of labels from PyTorch or None.
    """
    # TODO: implement the three-stage strategy.
    raise NotImplementedError("load_imagenet_labels not implemented")


def load_pretrained_resnet(name: str = "resnet50", device='cpu'):
    """
    Instantiate a torchvision ResNet architecture and return (model, preprocess[, weights_enum]).
    Students should:
      - attempt to use torchvision 'weights' API when available to get both model and transforms
      - otherwise construct a model with pretrained=False and provide a reasonable preprocess transform
    This skeleton returns a bare ResNet architecture (no pretrained weights) and a basic preprocess pipeline.
    """
    name = name.lower()
    if name not in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152"):
        raise ValueError(f"Unsupported model '{name}'. Choose one of resnet18/34/50/101/152")
    model_fn = getattr(torchvision.models, name)

    # Minimal working instantiation (no pretrained weights).
    # TODO: students should replace this with code that uses torchvision weights (if available)
    # and returns weights_enum and transforms when possible.
    try:
        # TODO: implement torchvision weights API usage here
        weights_enum = getattr(???)
        weights = weights_enum.???
        model = model_fn(weights=???).to(???)
        preprocess = ???.transforms()
    except Exception:
        # TODO: implement fallback to non-pretrained model and basic preprocess
        model = model_fn(???).to(???)
        preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        weights_enum = None

    model.eval()
    return model, preprocess, weights_enum


def model_summary(model: nn.Module):
    """
    Produce a small dictionary summarizing the model (parameter counts, number of conv/linear layers, etc).
    TODO: Students should fill out useful statistics (the skeleton returns a minimal structure).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # TODO: implement a full summary of the model.
    summary = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "conv_layers_count": ???,
        "conv_out_channels": ???,
        "batchnorm_count": ???,
        "linear_layers_count": ???,
        "layers_with_params_count": ???,
        "top_level_modules": [type(m).__name__ for m in model.children()],
    }
    return summary


def print_model_info(name: str, summary: dict, labels):
    """
    Nicely print model information and label availability.
    """
    print("="*80)
    print(f"Model: {name}")
    print(f"Top-level modules: {summary.get('top_level_modules')}")
    print(f"Total parameters: {summary.get('total_params'):,}")
    print(f"Trainable parameters: {summary.get('trainable_params'):,}")
    if labels is not None:
        print(f"ImageNet label count available: {len(labels)}")
    else:
        print("No ImageNet label metadata available locally and download failed; predictions will be class indices.")
    print("="*80)


def resolve_label_strings_to_indices(labels_list, request):
    """
    Resolve a user-requested label (string or integer) to ImageNet indices using the provided labels_list.
    Returns a list of indices (possibly empty).
    """
    if labels_list is None:
        return []
    out = []
    if isinstance(request, int):
        if 0 <= request < len(labels_list):
            out.append(request)
        return out
    req_norm = normalize_text(str(request))
    for i, lab in enumerate(labels_list):
        lab_norm = normalize_text(lab)
        if lab_norm == req_norm:
            out.append(i)
    # substring fallback
    if not out:
        for i, lab in enumerate(labels_list):
            lab_norm = normalize_text(lab)
            if req_norm in lab_norm or lab_norm in req_norm:
                out.append(i)
    return out


def build_mapping_from_folders_to_imagenet(folder_names, imagenet_labels, fuzzy_threshold=0.65, verbose=False):
    """
    Return dict: folder_name -> set(indices)
    Heuristics:
      - normalized exact match
      - token match (split on spaces)
      - substring match (either direction)
      - singularize folder name (naive) and try again
      - fuzzy match above threshold
    """
    mapping = {}
    label_norms = [normalize_text(l) for l in imagenet_labels] if imagenet_labels else []
    for folder in folder_names:
        candidates = set()
        f_norm = normalize_text(folder)
        f_sing = simple_singular(f_norm)

        # try exact normalized equality against any label
        if imagenet_labels:
            for idx, ln in enumerate(label_norms):
                if ln == f_norm or ln == f_sing:
                    candidates.add(idx)

            # try substring/token matches
            if not candidates:
                f_tokens = set(f_norm.split())
                for idx, ln in enumerate(label_norms):
                    ln_tokens = set(ln.split())
                    if f_norm in ln or ln in f_norm:
                        candidates.add(idx)
                    elif f_tokens & ln_tokens:
                        candidates.add(idx)

            # fuzzy fallback
            if not candidates:
                best_score = 0.0
                best_idx = None
                for idx, ln in enumerate(label_norms):
                    score = fuzzy_similarity(f_norm, ln)
                    if score > best_score:
                        best_score = score
                        best_idx = idx
                if best_score >= fuzzy_threshold and best_idx is not None:
                    candidates.add(best_idx)

        mapping[folder] = candidates
        if verbose:
            print(f"Mapping for '{folder}': {len(candidates)} candidates (sample indices: {sorted(list(candidates))[:5]})")
    return mapping


def predict_image(model, preprocess, image_path: Path, device='cpu'):
    """
    Run a forward pass on the given image and return (pred_idx, prob).
    Students should ensure preprocessing matches the model's expected transforms.
    """
    # TODO: Open the image, convert it to RGB, apply preprocess, move to device
    img = ???
    inp = ???
    with torch.no_grad():
        out = model(inp)
        probs = torch.nn.functional.softmax(out, dim=1)
        top_prob, top_idx = torch.topk(probs, k=1, dim=1)
    return int(top_idx[0,0].item()), float(top_prob[0,0].item())


def main(args):
    """
    Main orchestration: load model, labels, build mapping, run predictions over dataset,
    compute and print accuracy, and save the model state_dict.
    NOTE: The final save step is left as a TODO so students implement saving using torch.save.
    """
    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")

    device = 'cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu'
    print(f"Using device: {device}")

    # Load model (skeleton returns a non-pretrained architecture)
    model, preprocess = load_pretrained_resnet(args.model, device=device)
    imagenet_labels = load_imagenet_labels(args.model)
    if imagenet_labels is None:
        print("Warning: Could not obtain ImageNet labels. Provide --label-map to map folders to indices or enable internet so labels can be downloaded.")
    summary = model_summary(model)
    print_model_info(args.model, summary, imagenet_labels)

    # gather classes
    class_folders = [p for p in sorted(data_dir.iterdir()) if p.is_dir()]
    if not class_folders:
        raise RuntimeError(f"No subfolders found in {data_dir}. Each class should be in its own folder with images.")
    folder_names = [p.name for p in class_folders]

    # Load optional user-provided mapping file (JSON)
    user_map = {}
    if args.label_map:
        lm_path = Path(args.label_map)
        if not lm_path.exists():
            raise FileNotFoundError(f"Label map file {lm_path} not found")
        with open(lm_path, "r", encoding="utf-8") as f:
            user_map = json.load(f)
        print(f"Loaded user label map with {len(user_map)} entries (will try to resolve strings -> indices).")

    # Resolve user_map to indices if possible (safely, minimal skeleton)
    user_map_indices = {}
    if user_map:
        for k, v in user_map.items():
            indices = set()
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, int):
                        indices.add(item)
                    else:
                        if imagenet_labels:
                            resolved = resolve_label_strings_to_indices(imagenet_labels, item)
                            indices.update(resolved)
                user_map_indices[k] = indices
            else:
                if isinstance(v, int):
                    user_map_indices[k] = {v}
                else:
                    if imagenet_labels:
                        user_map_indices[k] = set(resolve_label_strings_to_indices(imagenet_labels, v))

    # Build automated mapping for all folders (minimal skeleton behavior)
    auto_map = {}
    if imagenet_labels:
        auto_map = build_mapping_from_folders_to_imagenet(folder_names, imagenet_labels, fuzzy_threshold=args.fuzzy_threshold, verbose=args.verbose)

    # final mapping: prioritize user_map_indices, otherwise auto_map
    final_map = {}
    for f in folder_names:
        if f in user_map_indices and user_map_indices[f]:
            final_map[f] = user_map_indices[f]
        else:
            final_map[f] = auto_map.get(f, set())

    # print mapping summary (minimal)
    print("="*80)
    print("Folder -> mapped ImageNet label indices summary (showing up to 6 indices, and sample labels if available):")
    for f in folder_names:
        idxs = sorted(list(final_map.get(f, set())))
        sample_labels = [imagenet_labels[i] for i in idxs[:6]] if imagenet_labels and idxs else []
        print(f"  {f} : {len(idxs)} indices -> {idxs[:6]}  labels_sample={sample_labels}")
    print("="*80)

    # collect all (image_path, true_label) pairs
    items = []
    per_class_counts = {}
    for cf in class_folders:
        imgs = list_image_files(cf)
        for im in imgs:
            items.append((im, cf.name))
        per_class_counts[cf.name] = len(imgs)
    total_images = len(items)
    if total_images == 0:
        raise RuntimeError("Found 0 images in dataset. Check that images are .jpg/.png etc.")

    print(f"Found {len(folder_names)} classes and {total_images} images.")
    print("Beginning classification...")

    correct = 0
    per_class_correct = defaultdict(int)
    t_start = time.time()

    for img_path, true_label in tqdm(items, desc="Classifying", unit="img"):
        pred_idx, prob = predict_image(model, preprocess, img_path, device=device)
        pred_label = imagenet_labels[pred_idx] if imagenet_labels and 0 <= pred_idx < len(imagenet_labels) else str(pred_idx)

        mapped_indices = final_map.get(true_label, set())

        matched = False
        reason = ""
        # NOTE: The detailed matching heuristics are intentionally left as TODO
        # for students to implement fully (including fuzzy fallbacks).
        # Minimal skeleton behavior: only exact mapped-index membership.
        if mapped_indices:
            if pred_idx in mapped_indices:
                matched = True
                reason = "mapped-index-match"
            else:
                matched = False
                reason = "mapped-miss"
        else:
            # no mapping: naive text comparison
            norm_true = normalize_text(true_label)
            norm_pred = normalize_text(pred_label)
            if norm_true == norm_pred or norm_true in norm_pred or norm_pred in norm_true:
                matched = True
                reason = "no-map-text-match"
            else:
                matched = False
                reason = "no-map-miss"

        if matched:
            correct += 1
            per_class_correct[true_label] += 1

        if args.verbose:
            print(f"{img_path.name} -> pred_idx={pred_idx} pred_label='{pred_label}' p={prob:.3f} | true='{true_label}' | matched={matched} ({reason})")

    t_end = time.time()
    total_time = t_end - t_start
    accuracy = correct / total_images * 100.0

    print("\n" + "="*80)
    print(f"Total images classified : {total_images}")
    print(f"Correct predictions      : {correct}")
    print(f"Overall accuracy         : {accuracy:.2f}%")
    print(f"Total classification time: {total_time:.2f} seconds")
    print(f"Avg time / image         : {total_time/total_images:.4f} seconds")
    print("="*80)
    print("Per-class summary (class : correct / total -> accuracy):")
    for cls in sorted(per_class_counts.keys()):
        c = per_class_correct.get(cls, 0)
        tot = per_class_counts[cls]
        acc = (c / tot * 100.0) if tot>0 else 0.0
        print(f"  {cls} : {c} / {tot} -> {acc:.2f}%")
    print("="*80)
    if imagenet_labels is None:
        print("Warning: ImageNet label strings unavailable (download failed). Provide --label-map or enable internet.")
    print("Done.")

    # TODO: save model state_dict to disk so unit tests can load it


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify images in folder structure using pretrained ResNet (ImageNet weights). Improved mapping to ImageNet labels.")
    parser.add_argument("--data-dir", type=str, default="../datasets/animals", help="Path to dataset root (folders of classes)")
    parser.add_argument("--model", type=str, default="resnet50", help="ResNet variant: resnet18, resnet34, resnet50, resnet101, resnet152")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.65, help="Fuzzy match threshold (0-1) for mapping and fallback matching.")
    parser.add_argument("--label-map", type=str, default="", help="Optional JSON file mapping folder names to ImageNet labels/indices.")
    parser.add_argument("--verbose", action="store_true", help="Print each prediction (verbose).")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU even if CUDA available.")
    args = parser.parse_args()
    main(args)