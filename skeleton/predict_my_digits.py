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
from typing import List
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from improved_digit_cnn import CNN


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
    #   - automatically invert if background is light
    #   - denoise with Gaussian blur and normalize contrast
    #   - apply Otsu thresholding / binarization
    #   - find digit bounding box (cv2.findNonZero + cv2.boundingRect) with a fallback center crop
    #   - pad to square and resize to 28x28 preserving aspect ratio
    #   - convert to float32 array in [0,1], apply normalization (arr - 0.1307)/0.3081
    #   - if show=True, display the preprocessed image for debugging
    #   - return a torch tensor of shape (1,1,28,28) on the requested device
    raise NotImplementedError("image_to_mnist_tensor: TODO implement preprocessing pipeline")


def load_trained_model(model_path: str, device: torch.device):
    """
    Load state_dict into CNN. Uses non-strict load and prints a warning if any keys differ.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # TODO: Instantiate CNN(), load the state_dict from model_path (map_location=device),
    #       attempt a strict load and if that fails try non-strict load and print a helpful warning.
    #       Ensure model.eval() is called before returning.
    raise NotImplementedError("load_trained_model: TODO implement model loading logic")


def predict_images(model_path: str, image_paths: List[str], device_str: str = "cpu", show=False):
    device = torch.device(device_str)

    # TODO: load the model using load_trained_model, preprocess each image using image_to_mnist_tensor,
    #       run a forward pass, compute softmax/confidence if possible, and format result strings.
    #
    # Expected output format per image:
    #   "Prediction for <basename>: <digit>"  OR
    #   "Prediction for <basename>: <digit> (conf=<probability>)"
    raise NotImplementedError("predict_images: TODO implement prediction loop")


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
