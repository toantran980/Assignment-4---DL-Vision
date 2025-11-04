# test_improved_digit_cnn.py
import os
import unittest
import inspect
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import the student's training script/module
import improved_digit_cnn as mod


class TestImprovedDigitCNN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Accept either filename (students may have slight naming differences).
        candidates = [Path("improved_digit_cnn_.pth"), Path("improved_digit_cnn.pth")]
        found = None
        for p in candidates:
            if p.exists():
                found = p
                break
        cls.pth_path = found

        # Prepare MNIST test dataset and DataLoader (same normalization as student's script)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        # download may write into current directory; that's fine
        cls.test_ds = datasets.MNIST(root='.', train=False, download=True, transform=transform)
        # Use a relatively large batch for speed but small enough for CPU memory
        cls.test_loader = DataLoader(cls.test_ds, batch_size=512, shuffle=False, num_workers=0, pin_memory=False)

        # device for loading model weights (we'll load to CPU for portability)
        cls.device = torch.device('cpu')

    def test_required_functions_exist(self):
        """Verify that required names exist in the module."""
        required = ["CNN", "set_seed", "train_one_epoch", "evaluate", "main"]
        for name in required:
            self.assertTrue(hasattr(mod, name),
                            msg=f"Required object '{name}' missing from improved_digit_cnn.py")

    def test_device_selection_present(self):
        """Confirm the script includes device selection logic (supports CPU/GPU)."""
        # Inspect source of the module (main in particular) for expected strings
        src = inspect.getsource(mod)
        # We look for either torch.device(...) usage or torch.cuda.is_available() check
        has_torch_device = "torch.device" in src
        has_cuda_check = "torch.cuda.is_available" in src or "torch.cuda.is_available()" in src
        self.assertTrue(has_torch_device or has_cuda_check,
                        msg="improved_digit_cnn.py does not appear to include device selection logic (torch.device / torch.cuda.is_available).")

    def test_pth_file_present(self):
        """The exported model state_dict file must exist."""
        self.assertIsNotNone(self.pth_path, msg="No model .pth file found. Expected improved_digit_cnn_.pth or improved_digit_cnn.pth in the working dir.")

    def test_model_load_and_accuracy(self):
        """
        Load the saved state_dict into CNN and evaluate on the MNIST test set.
        Requires accuracy >= 0.99 (99%).
        """
        if self.pth_path is None:
            self.skipTest("Model .pth not found; skipping accuracy test.")

        # Instantiate model architecture from student module
        self.assertTrue(hasattr(mod, "CNN"), "CNN class missing from improved_digit_cnn.py")
        model = mod.CNN()
        model.to(self.device)
        model.eval()

        # Load state dict (be tolerant to GPU-saved tensors by mapping to cpu)
        try:
            sd = torch.load(str(self.pth_path), map_location='cpu')
        except Exception as e:
            self.fail(f"Failed to load state_dict from {self.pth_path}: {e}")

        # load into model non-strictly to be a little tolerant to extra keys
        try:
            model.load_state_dict(sd, strict=False)
        except Exception as e:
            self.fail(f"Loading state_dict into CNN failed: {e}")

        # Use student's evaluate function when possible for consistency
        self.assertTrue(hasattr(mod, "evaluate"), "evaluate function missing from improved_digit_cnn.py")
        # Evaluate on test_loader (forward-only)
        with torch.no_grad():
            acc = mod.evaluate(model, self.test_loader, self.device)
        # accuracy returned as fraction (0..1)
        self.assertIsInstance(acc, float, "evaluate(...) should return a float accuracy value.")
        # Check threshold: >= 0.99
        min_acc = 0.99
        self.assertGreaterEqual(acc, min_acc, msg=f"Model accuracy {acc:.4f} is below required threshold {min_acc:.4f} (99%).")

    def test_evaluate_returns_valid_range(self):
        """Sanity check: evaluate returns a value in [0,1]."""
        # instantiate a fresh model and run evaluate but we will not fail the test if pth missing
        model = mod.CNN()
        model.to(self.device)
        # random init; evaluate on small subset to be quick
        loader = DataLoader(self.test_ds, batch_size=128, shuffle=False, num_workers=0)
        with torch.no_grad():
            acc = mod.evaluate(model, loader, self.device)
        self.assertIsInstance(acc, float)
        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 1.0)


if __name__ == "__main__":
    unittest.main()
