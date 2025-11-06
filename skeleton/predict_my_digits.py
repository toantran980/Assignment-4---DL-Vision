#!/usr/bin/env python3

"""
predict_my_digits.py

Predict digit images using a CNN trained on MNIST (improved_digit_cnn.pth).
Automatically inverts and normalizes images to match MNIST style.

Usage:
  python predict_my_digits.py --model improved_digit_cnn.pth --images digit2.jpg digit4.jpg digit8.jpg
"""

import argparse
import os
import sys
from typing import List
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from skeleton.improved_digit_cnn import CNN


def image_to_mnist_tensor(path: str, device: torch.device, show=False):
    """
    Convert a photo of a handwritten digit into an MNIST-style tensor.
    NOTE: normalization matches training: transforms.Normalize((0.1307,), (0.3081,))
    """
    import cv2

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    # TODO: Implement the full preprocessing pipeline described in the original script:
    #   - load image as grayscale (PIL or cv2) and convert to numpy array
    pil_img = Image.open(path).convert("L")
    img = np.array(pil_img)

    #   - automatically invert if background is light
    if np.mean(img) > 127:
        img = 255 - img
        
    #   - denoise with Gaussian blur and normalize contrast
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    #   - apply Otsu thresholding / binarization
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #   - find digit bounding box (cv2.findNonZero + cv2.boundingRect) with a fallback center crop
    coords = cv2.findNonZero(img)

    #   - pad to square and resize to 28x28 preserving aspect ratio
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit = img[y:y+h, x:x+w]
    else:
        h, w = img.shape
        min_dim = min(h, w)
        digit = img[(h - min_dim)//2:(h + min_dim)//2, (w - min_dim)//2:(w + min_dim)//2]
        h, w = digit.shape
    size = max(h, w)
    square_img = np.zeros((size, size), dtype=np.uint8)
    square_img[(size - h)//2:(size - h)//2 + h, (size - w)//2:(size - w)//2 + w] = digit
    resized_img = cv2.resize(square_img, (28, 28), interpolation=cv2.INTER_AREA)

    #   - convert to float32 array in [0,1], apply normalization (arr - 0.1307)/0.3081
    norm_img = (resized_img.astype(np.float32) / 255.0 - 0.1307) / 0.3081

    #   - if show=True, display the preprocessed image for debugging
    if show:
        cv2.imshow("Preprocessed Image", resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    #   - return a torch tensor of shape (1,1,28,28) on the requested device
    tensor = torch.tensor(norm_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    return tensor
    #raise NotImplementedError("image_to_mnist_tensor: TODO implement preprocessing pipeline")


def load_trained_model(model_path: str, device: torch.device):
    """
    Load state_dict into CNN. Uses non-strict load and prints a warning if any keys differ.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # TODO: Instantiate CNN(), load the state_dict from model_path (map_location=device),
    #       attempt a strict load and if that fails try non-strict load and print a helpful warning.
    #       Ensure model.eval() is called before returning.
    model = CNN().to(device)
    state_dict = torch.load(model_path, map_location=device)
    
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Warning: Non-strict loading due to key mismatch: {e}")
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    return model
    #raise NotImplementedError("load_trained_model: TODO implement model loading logic")


def predict_images(model_path: str, image_paths: List[str], device_str: str = "cpu", show=False):
    device = torch.device(device_str)

    # TODO: load the model using load_trained_model, preprocess each image using image_to_mnist_tensor,
    #       run a forward pass, compute softmax/confidence if possible, and format result strings.
    #
    # Expected output format per image:
    #   "Prediction for <basename>: <digit>"  OR
    #   "Prediction for <basename>: <digit> (conf=<probability>)"
    model = load_trained_model(model_path, device)
    results = []

    with torch.no_grad():
        for image_path in image_paths:
            # Preprocess image
            tensor = image_to_mnist_tensor(image_path, device, show=show)
            
            # Get model prediction
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1).item()
            confidence = probs[0][prediction].item()
            
            # Format result string
            basename = os.path.basename(image_path)
            result = f"Prediction for {basename}: {prediction} (conf={confidence:.3f})"
            results.append(result)
            print(result)
    
    return results
    #raise NotImplementedError("predict_images: TODO implement prediction loop")


def main():
    parser = argparse.ArgumentParser(description="Predict digit images using improved_digit_cnn.pth.")
    parser.add_argument("--model", type=str, default="improved_digit_cnn.pth", help="Path to model file")
    parser.add_argument(
        "--images",
        nargs="*",
        default=["../datasets/digits/digit2.jpg", "../datasets/digits/digit4.jpg", "../datasets/digits/digit6.jpg", "../datasets/digits/digit8.jpg"],
        help="Image files to predict",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on (cpu or cuda)")
    parser.add_argument("--show", action="store_true", help="Show preprocessed images for debugging")
    args = parser.parse_args()

    lines = predict_images(args.model, args.images, args.device, show=args.show)
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
