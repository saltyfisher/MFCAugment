import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def project_on_path(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))


def import_fresh(module_name):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_utils_import_does_not_require_graph_or_cv_modules(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.setitem(sys.modules, "gco", None)

    utils = import_fresh("core.utils")

    assert hasattr(utils, "KL_loss")
    assert hasattr(utils, "kl_divergence_multivariate_torch")


def test_trimap_stats_branch_requires_cv2_only_when_used(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    utils = import_fresh("core.utils")

    with pytest.raises(ImportError, match="cv2"):
        utils.trimap_generate(
            np.zeros((2, 2, 3), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            trimap_gen="stats",
        )


def test_trimap_graph_branch_requires_gco_only_when_used(monkeypatch):
    monkeypatch.setitem(sys.modules, "gco", None)
    utils = import_fresh("core.utils")

    with pytest.raises(ImportError, match="gco"):
        utils.trimap_generate(
            np.zeros((2, 2, 3), dtype=np.float32),
            np.full((2, 2), 0.5, dtype=np.float32),
            trimap_gen="graph",
            sigma1=1.0,
        )


def test_kl_torch_loss_is_near_zero_for_identical_samples():
    from core import utils

    samples = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float32)

    loss = utils.kl_divergence_multivariate_torch(
        samples,
        samples,
        xbins=32,
        noise_level=0.0,
    )

    assert loss == pytest.approx(0.0, abs=1e-6)


def test_fastaa_augment_registry_and_level_mapping(monkeypatch):
    from core import augmentations_fastaa as fastaa

    calls = []

    def fake_op(img, value):
        calls.append((img.copy(), value))
        return img

    monkeypatch.setitem(fastaa.AUGMENT_REGISTRY, "Fake", (fake_op, -2, 2))
    image = Image.new("RGB", (2, 2))

    result = fastaa.apply_augment(image, "Fake", 0.75)

    assert result.size == image.size
    assert calls[0][1] == pytest.approx(1.0)


def test_fastaa_augmentation_class_applies_policy(monkeypatch):
    from core import augmentations_fastaa as fastaa

    image = Image.new("RGB", (2, 2), color="white")
    calls = []

    def fake_op(img, value):
        calls.append(value)
        return img

    policies = [[("Fake", 1.0, 0.25)]]
    aug = fastaa.Augmentation(policies)
    monkeypatch.setitem(aug.augmentations, "Fake", (fake_op, 0, 8))

    result = aug(image)

    assert result.size == image.size
    assert calls == [2.0]


def test_myaugment_without_probability_dimension_applies_all_policy_ops(monkeypatch):
    from core import augmentations

    calls = []

    def fake_apply(img, op_name, magnitude, interpolation, fill):
        calls.append((op_name, magnitude))
        return img

    monkeypatch.setattr(augmentations, "_apply_op", fake_apply)

    policy = {
        "op_index": np.array([[0, 1]]),
        "magnitude_index": np.array([[0, 1]]),
        "prob_index": [],
    }
    aug_space = {
        "OpA": (torch.tensor([0.5]), torch.tensor([0.0]), False),
        "OpB": (torch.tensor([0.0, 1.5]), torch.tensor([0.0]), False),
    }
    augmenter = augmentations.MyAugment(policy, num_ops=2)

    result = augmenter.augment_img("image", aug_space, fill=None)

    assert result == "image"
    assert calls == [("OpA", 0.5), ("OpB", 1.5)]
