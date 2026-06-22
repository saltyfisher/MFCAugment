import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def project_on_path(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))


def import_fresh(module_name):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_data_import_does_not_require_cv2_or_gco(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.setitem(sys.modules, "gco", None)
    sys.modules.pop("core.augmentations", None)

    data = import_fresh("data")

    assert data.get_resize_size("breakhis") == (448, 448)
    assert data.get_resize_size("chestct") == (224, 224)


def test_augmentations_import_does_not_require_cv2_or_gco(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.setitem(sys.modules, "gco", None)

    augmentations = import_fresh("core.augmentations")

    assert hasattr(augmentations, "MyRandAugment")
    assert hasattr(augmentations, "MyAugment")


def test_mfcaugment_import_does_not_require_optional_search_dependencies(monkeypatch):
    optional_modules = [
        "cv2",
        "gco",
        "cvxpy",
        "pyDOE3",
        "kornia",
        "tensorboardX",
        "torch_pca",
        "hyperopt",
    ]
    for module_name in optional_modules:
        monkeypatch.setitem(sys.modules, module_name, None)
    sys.modules.pop("core.MFCAugment", None)

    mfc = import_fresh("core.MFCAugment")

    assert hasattr(mfc, "MFCAugment")
    assert hasattr(mfc, "formatPolicy")


def test_dataset_roots_are_explicit():
    import data

    roots = data.get_dataset_roots(Path("root"), "breakhis", "100")
    assert roots == (Path("root") / "BreakHis" / "100", None)

    roots = data.get_dataset_roots(Path("root"), "chestct", None)
    assert roots == (
        Path("root") / "chest-ctscan-images_datasets" / "train",
        Path("root") / "chest-ctscan-images_datasets" / "test",
    )

    roots = data.get_dataset_roots(Path("root"), "corona", None)
    assert roots == (
        Path("root") / "Coronahack-Chest-XRay-Dataset" / "train",
        Path("root") / "Coronahack-Chest-XRay-Dataset" / "test",
    )


def test_mydata_get_labels_uses_imagefolder_targets():
    import data

    dataset = data.Mydata.__new__(data.Mydata)
    dataset.targets = [0, 1, 1]

    assert dataset.get_labels() == [0, 1, 1]


def test_validation_split_returns_full_training_dataset():
    import data

    class FakeDataset:
        def get_labels(self):
            return [0, 0, 1, 1]

        def __len__(self):
            return 4

    result = data.split_train_val_dataset(
        FakeDataset(),
        train_transform="train",
        test_transform="test",
        validation_folds=2,
        random_state=42,
    )

    full_dataset, train_folds, val_folds = result
    assert isinstance(full_dataset, FakeDataset)
    assert len(train_folds) == 2
    assert len(val_folds) == 2
    assert all(fold.transform == "train" for fold in train_folds)
    assert all(fold.transform == "test" for fold in val_folds)
