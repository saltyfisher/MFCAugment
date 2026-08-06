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
    assert data.get_resize_size("cifar-fs") == (32, 32)
    assert data.get_resize_size("miniimagenet") == (84, 84)
    assert data.get_resize_size("pad-ufes-20") == (320, 320)


def test_natural_image_datasets_use_dataset_normalization(monkeypatch):
    import data

    class FakeTransforms:
        class Resize:
            def __init__(self, size):
                self.size = size

        class ToTensor:
            pass

        class Normalize:
            def __init__(self, mean, std):
                self.mean = mean
                self.std = std

        class Compose:
            def __init__(self, transforms):
                self.transforms = transforms

    monkeypatch.setattr(data, "load_transforms", lambda: FakeTransforms)

    cifar_transform = data.build_base_transform((32, 32), "cifar-fs")
    mini_transform = data.build_base_transform((84, 84), "miniimagenet")
    medical_transform = data.build_base_transform((224, 224), "chestct")

    cifar_normalize = cifar_transform.transforms[-1]
    mini_normalize = mini_transform.transforms[-1]

    assert cifar_normalize.__class__.__name__ == "Normalize"
    assert cifar_normalize.mean == pytest.approx((0.5071, 0.4867, 0.4408))
    assert cifar_normalize.std == pytest.approx((0.2675, 0.2565, 0.2761))
    assert mini_normalize.__class__.__name__ == "Normalize"
    assert mini_normalize.mean == pytest.approx((0.485, 0.456, 0.406))
    assert mini_normalize.std == pytest.approx((0.229, 0.224, 0.225))
    assert [step.__class__.__name__ for step in medical_transform.transforms] == [
        "Resize",
        "ToTensor",
    ]


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
        Path("root") / "chest-ctscan-images_datasets" / "Data" / "train",
        Path("root") / "chest-ctscan-images_datasets" / "Data" / "test",
    )


def test_chestct_roots_accept_direct_data_directory(tmp_path):
    import data

    data_root = tmp_path / "Data"
    (data_root / "train").mkdir(parents=True)
    (data_root / "test").mkdir()

    assert data.get_dataset_roots(data_root, "chestct", None) == (
        data_root / "train",
        data_root / "test",
    )


def test_breakhis_roots_accept_direct_dataset_directory(tmp_path):
    import data

    dataset_root = tmp_path / "BreakHis"
    (dataset_root / "40").mkdir(parents=True)

    assert data.get_dataset_roots(dataset_root, "breakhis", "40") == (
        dataset_root / "40",
        None,
    )

    roots = data.get_dataset_roots(Path("root"), "corona", None)
    assert roots == (
        Path("root") / "Coronahack-Chest-XRay-Dataset" / "train",
        Path("root") / "Coronahack-Chest-XRay-Dataset" / "test",
    )


def make_imagefolder(root, class_names=("class_a", "class_b"), images_per_class=4):
    for class_name in class_names:
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for index in range(images_per_class):
            (class_dir / f"image_{index}.png").write_bytes(b"")


def test_few_shot_imagefolder_roots_accept_repo_and_direct_directories(tmp_path):
    import data

    cifar_root = tmp_path / "cifar-fs" / "cifar100" / "data"
    mini_root = tmp_path / "miniimagenet" / "data"
    make_imagefolder(cifar_root)
    make_imagefolder(mini_root)

    assert data.get_dataset_roots(tmp_path, "cifar-fs", None) == (cifar_root, None)
    assert data.get_dataset_roots(tmp_path / "cifar-fs" / "cifar100", "cifar-fs", None) == (
        cifar_root,
        None,
    )
    assert data.get_dataset_roots(cifar_root, "cifar-fs", None) == (cifar_root, None)
    assert data.get_dataset_roots(tmp_path, "miniimagenet", None) == (mini_root, None)
    assert data.get_dataset_roots(mini_root, "miniimagenet", None) == (mini_root, None)


def test_pad_ufes_roots_accept_organized_and_train_test_directories(tmp_path):
    import data

    organized_root = tmp_path / "PAD-UFES-20" / "organized_dataset"
    make_imagefolder(organized_root)

    assert data.get_dataset_roots(tmp_path, "pad-ufes-20", None) == (organized_root, None)
    assert data.get_dataset_roots(organized_root, "pad-ufes-20", None) == (organized_root, None)

    split_root = tmp_path / "split_pad"
    make_imagefolder(split_root / "train")
    make_imagefolder(split_root / "test")

    assert data.get_dataset_roots(split_root, "pad-ufes-20", None) == (
        split_root / "train",
        split_root / "test",
    )


def test_pad_ufes_roots_accept_raw_metadata_directory(tmp_path):
    import data

    raw_root = tmp_path / "PAD-UFES-20"
    (raw_root / "images").mkdir(parents=True)
    (raw_root / "images" / "sample.png").write_bytes(b"")
    (raw_root / "metadata.csv").write_text("img_id,diagnostic\nsample.png,BCC\n")

    assert data.get_dataset_roots(tmp_path, "pad-ufes-20", None) == (raw_root, None)
    assert data.get_dataset_roots(raw_root, "pad-ufes-20", None) == (raw_root, None)


def test_pad_ufes_dataset_reads_metadata_samples(tmp_path):
    import data

    raw_root = tmp_path / "PAD-UFES-20"
    image_root = raw_root / "images"
    image_root.mkdir(parents=True)
    (image_root / "ack.png").write_bytes(b"")
    (image_root / "mel.png").write_bytes(b"")
    (raw_root / "metadata.csv").write_text(
        "img_id,diagnostic\nack.png,ACK\nmel.png,MEL\n",
        encoding="utf-8",
    )

    dataset = data.PadUfes20Dataset(raw_root, transform=lambda image: f"transformed-{image}", loader=lambda path: Path(path).name)

    assert dataset.classes == ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
    assert dataset.class_to_idx["ACK"] == 0
    assert dataset.class_to_idx["MEL"] == 2
    assert dataset.targets == [0, 2]
    assert dataset[1] == ("transformed-mel.png", 2, 1)


def test_get_data_reads_pad_ufes_from_raw_metadata_directory(tmp_path, monkeypatch):
    import data

    raw_root = tmp_path / "PAD-UFES-20"
    image_root = raw_root / "images"
    image_root.mkdir(parents=True)
    for class_name in data.PAD_UFES_20_CLASSES:
        for index in range(2):
            (image_root / f"{class_name}_{index}.png").write_bytes(b"")
    (raw_root / "metadata.csv").write_text(
        "img_id,diagnostic\n"
        + "\n".join(f"{class_name}_{index}.png,{class_name}" for class_name in data.PAD_UFES_20_CLASSES for index in range(2))
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(data, "build_transforms", lambda strategy, resize_size, dataset=None: ("train-transform", "test-transform"))

    traintest_dataset, test_dataset, resize_size, train_transform = data.get_data(
        strategy="",
        dataset="pad-ufes-20",
        magnification=None,
        dataroot=tmp_path,
        random_state=42,
        test_split=0.5,
    )

    assert resize_size == (320, 320)
    assert train_transform == "train-transform"
    assert len(traintest_dataset) == 6
    assert len(test_dataset) == 6
    assert set(traintest_dataset.get_labels()) == set(range(6))
    assert set(test_dataset.get_labels()) == set(range(6))


def test_get_data_reads_few_shot_imagefolder_dataset_from_resolved_root(tmp_path, monkeypatch):
    import data

    data_root = tmp_path / "miniimagenet" / "data"
    make_imagefolder(data_root, images_per_class=4)
    captured = {}

    class FakeDataset:
        def __init__(self, root, transform):
            captured["root"] = root
            captured["transform"] = transform
            self.samples = [("a", 0), ("b", 0), ("c", 1), ("d", 1)]

    def fake_split(full_dataset, train_transform, test_transform, test_split, random_state):
        captured["split"] = (train_transform, test_transform, test_split, random_state)
        return "train-dataset", "test-dataset"

    monkeypatch.setattr(data, "build_transforms", lambda strategy, resize_size, dataset=None: ("train-transform", "test-transform"))
    monkeypatch.setattr(data, "Mydata", FakeDataset)
    monkeypatch.setattr(data, "split_imagefolder_train_test", fake_split)

    traintest_dataset, test_dataset, resize_size, train_transform = data.get_data(
        strategy="",
        dataset="miniimagenet",
        magnification=None,
        dataroot=tmp_path,
        random_state=42,
        test_split=0.5,
    )

    assert captured["root"] == str(data_root)
    assert captured["transform"] == "train-transform"
    assert captured["split"] == ("train-transform", "test-transform", 0.5, 42)
    assert resize_size == (84, 84)
    assert train_transform == "train-transform"
    assert traintest_dataset == "train-dataset"
    assert test_dataset == "test-dataset"


def test_split_imagefolder_train_test_keeps_each_class_in_both_splits():
    import data

    class FakeDataset:
        samples = [(f"image-{index}", index // 4) for index in range(8)]
        targets = [label for _, label in samples]
        transform = "base-transform"
        target_transform = None

        def __len__(self):
            return len(self.samples)

        @staticmethod
        def loader(path):
            return path

    train_dataset, test_dataset = data.split_imagefolder_train_test(
        FakeDataset(),
        train_transform="train-transform",
        test_transform="test-transform",
        test_split=0.5,
        random_state=42,
    )

    assert len(train_dataset) == 4
    assert len(test_dataset) == 4
    assert set(train_dataset.get_labels()) == {0, 1}
    assert set(test_dataset.get_labels()) == {0, 1}
    assert train_dataset.transform == "train-transform"
    assert test_dataset.transform == "test-transform"


def test_num_class_includes_few_shot_datasets():
    networks = import_fresh("networks")

    assert networks.num_class("cifar-fs") == 100
    assert networks.num_class("miniimagenet") == 100
    assert networks.num_class("pad-ufes-20") == 6


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


def test_mydatasubset_initializes_transform_attributes():
    import data

    class FakeDataset:
        samples = [("image-path", 1)]
        transform = None
        target_transform = None

        @staticmethod
        def loader(path):
            return "image"

    subset = data.Mydatasubset(FakeDataset(), [0])

    assert subset[0] == ("image", 1, 0)
    assert subset.transform is None
    assert subset.target_transform is None


def test_mydatasubset_returns_subset_relative_index():
    import data

    class FakeDataset:
        samples = [(f"image-{i}", i % 2) for i in range(8)]
        transform = None
        target_transform = None

        @staticmethod
        def loader(path):
            return path

    subset = data.Mydatasubset(FakeDataset(), [5, 7])

    assert subset[0][2] == 0
    assert subset[1][2] == 1
