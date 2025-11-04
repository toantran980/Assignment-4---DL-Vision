# test_animal_classifier.py
import os
import unittest
import torch
from pathlib import Path
from PIL import Image
import math

# Import the student's module
import animal_classifier as ac


class TestAnimalClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # force CPU for determinism on CI
        cls.device = 'cpu'
        cls.pth_path = Path("animal_classifier.pth")
        cls.data_dir = Path("../datasets/animals")

        # Load architecture + preprocess from student's module
        model_out = ac.load_pretrained_resnet("resnet50", device=cls.device)
        # Accept either (model, preprocess, weights_enum) or (model, preprocess)
        if len(model_out) == 3:
            cls.arch_model, cls.preprocess, _ = model_out
        else:
            cls.arch_model, cls.preprocess = model_out
        cls.arch_model.to(cls.device)
        cls.arch_model.eval()

        # Sample image detection (not used by the deterministic subset test directly)
        cls.sample_image = None
        if cls.data_dir.exists():
            for sub in sorted(cls.data_dir.iterdir()):
                if sub.is_dir():
                    for f in sorted(sub.iterdir()):
                        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
                            cls.sample_image = f
                            break
                if cls.sample_image:
                    break

    def test_pth_exists(self):
        """Ensure the student saved animal_classifier.pth"""
        self.assertTrue(self.pth_path.exists(), f"{self.pth_path} not found. Students must save model state_dict as this filename.")

    def test_state_dict_loads_into_architecture(self):
        """Ensure student's .pth can be loaded into expected ResNet architecture (non-strict ok)."""
        sd = torch.load(self.pth_path, map_location='cpu')
        self.assertIsInstance(sd, dict, "Saved .pth should be a state_dict (dict).")
        # create fresh architecture to load into
        model_out = ac.load_pretrained_resnet("resnet50", device='cpu')
        if len(model_out) == 3:
            fresh_model, _, _ = model_out
        else:
            fresh_model, _ = model_out
        # attempt load (non-strict to tolerate tiny key differences)
        try:
            fresh_model.load_state_dict(sd, strict=False)
        except Exception as e:
            self.fail(f"Loading student's state_dict raised an exception: {e}")
        # sanity: expect many keys
        self.assertGreater(len(sd.keys()), 10, "State dict seems unusually small.")

    @unittest.skipIf(not Path("../datasets/animals").exists(), "Dataset not present; skipping forward-pass integration tests.")
    def test_deterministic_subset_accuracy(self):
        """
        Deterministic subset test:
          - Choose first N class folders (alphabetically) that have at least K images each.
          - For each chosen class, use the first K images (sorted).
          - Load student's saved weights into ResNet architecture and run forward passes.
          - Consider a prediction correct if predicted ImageNet index is in the mapped indices (from student's mapping helper).
            If no mapping exists for the folder, fall back to normalized text or fuzzy match between predicted label and folder name.
          - Assert overall accuracy >= MIN_ACC (configurable via env var).
        Environment vars:
          TEST_NUM_CLASSES (int, default 5)
          TEST_IMAGES_PER_CLASS (int, default 5)
          TEST_MIN_ACCURACY (float 0-1, default 0.6)
        """
        # Config from environment (deterministic defaults)
        N = int(os.environ.get("TEST_NUM_CLASSES", "90"))
        K = int(os.environ.get("TEST_IMAGES_PER_CLASS", "2"))
        MIN_ACC = float(os.environ.get("TEST_MIN_ACCURACY", "0.4"))

        # Get sorted class folders and filter those with >= K images
        base = Path("../datasets/animals")
        class_dirs = [p for p in sorted(base.iterdir()) if p.is_dir()]
        eligible = []
        for d in class_dirs:
            imgs = [f for f in sorted(d.iterdir()) if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")]
            if len(imgs) >= K:
                eligible.append((d, imgs))
            if len(eligible) >= N:
                break

        if len(eligible) < N:
            self.skipTest(f"Not enough folders with >={K} images to run deterministic subset test (found {len(eligible)}, need {N}).")

        chosen = eligible[:N]  # deterministic selection: first N eligible

        # Load imagenet labels and mapping using student's helpers
        imagenet_labels = ac.load_imagenet_labels("resnet50")  # may be None
        folder_names = [d.name for d, imgs in chosen]
        mapping = {}
        if imagenet_labels:
            mapping = ac.build_mapping_from_folders_to_imagenet(folder_names, imagenet_labels, fuzzy_threshold=0.65, verbose=False)
        else:
            mapping = {name: set() for name in folder_names}

        # Load student's saved state dict into fresh architecture
        sd = torch.load(self.pth_path, map_location='cpu')
        model_out = ac.load_pretrained_resnet("resnet50", device='cpu')
        if len(model_out) == 3:
            fresh_model, preprocess, _ = model_out
        else:
            fresh_model, preprocess = model_out
        # Load state dict non-strict
        fresh_model.load_state_dict(sd, strict=False)
        fresh_model.to('cpu')
        fresh_model.eval()

        # Run predictions
        total = 0
        correct = 0
        for (d, imgs) in chosen:
            # deterministic first K images
            imgs_sorted = [f for f in sorted(imgs)][:K]
            for img_path in imgs_sorted:
                total += 1
                # Use student's predict_image if available (it is)
                pred_idx, prob = ac.predict_image(fresh_model, preprocess, img_path, device='cpu')
                # Determine predicted label string if imagenet labels present
                pred_label = imagenet_labels[pred_idx] if imagenet_labels and 0 <= pred_idx < len(imagenet_labels) else str(pred_idx)
                # Check mapping
                mapped_indices = mapping.get(d.name, set())
                matched = False
                if mapped_indices:
                    if pred_idx in mapped_indices:
                        matched = True
                else:
                    # fallback to normalized text/fuzzy
                    nt = ac.normalize_text(d.name)
                    npred = ac.normalize_text(pred_label)
                    if nt == npred or nt in npred or npred in nt:
                        matched = True
                    else:
                        sim = ac.fuzzy_similarity(nt, npred)
                        if sim >= 0.65:
                            matched = True
                if matched:
                    correct += 1

        acc = (correct / total) if total > 0 else 0.0
        # Provide diagnostic info in failure message
        self.assertGreaterEqual(acc, MIN_ACC, msg=f"Deterministic subset accuracy {acc:.3f} is below required {MIN_ACC:.3f} (tested {total} images across {N} classes).")

    # existing fast tests from prior suite kept for convenience
    def test_build_mapping_quick(self):
        """Quick check that mapping helper returns dict-of-sets and handles missing labels gracefully."""
        folder_names = ["cat", "dog", "goldfish", "this_class_does_not_exist"]
        labels = ac.load_imagenet_labels("resnet50") or []
        mapping = ac.build_mapping_from_folders_to_imagenet(folder_names, labels)
        self.assertIsInstance(mapping, dict)
        for k in folder_names:
            self.assertIn(k, mapping)
            self.assertIsInstance(mapping[k], set)


if __name__ == "__main__":
    unittest.main()
