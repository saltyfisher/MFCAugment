import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_foundation_features as tff


def test_parser_defaults_to_dinov2_pad_features():
    args = tff.build_parser().parse_args([])

    assert args.dataset == "pad-ufes-20"
    assert args.feature_backend == "auto"
    assert args.feature_model == "facebook/dinov2-base"
    assert args.standardize_features is True
    assert args.feature_cache is True


@pytest.mark.parametrize(
    ("preset", "backend", "model"),
    [
        ("biomedclip", "openclip", "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"),
        ("derm_foundation", "keras", "google/derm-foundation"),
        ("chief", "chief", "CHIEF_CTransPath"),
        ("uni", "timm", "hf-hub:MahmoodLab/UNI"),
        ("medimageinsight", "medimageinsight", "lion-ai/MedImageInsights"),
    ],
)
def test_model_presets_select_specialized_backends(preset, backend, model):
    args = tff.apply_model_preset(tff.build_parser().parse_args(["--model_preset", preset]))

    assert args.feature_backend == backend
    assert args.feature_model == model


def test_create_feature_extractor_routes_optional_backends(monkeypatch):
    calls = []

    class FakeExtractor:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    monkeypatch.setattr(tff, "OpenCLIPFeatureExtractor", FakeExtractor)
    monkeypatch.setattr(tff, "TimmFeatureExtractor", FakeExtractor)
    monkeypatch.setattr(tff, "KerasEmbeddingExtractor", FakeExtractor)
    monkeypatch.setattr(tff, "ChiefFeatureExtractor", FakeExtractor)
    monkeypatch.setattr(tff, "MedImageInsightFeatureExtractor", FakeExtractor)

    args = tff.apply_model_preset(tff.build_parser().parse_args(["--model_preset", "biomedclip"]))
    assert isinstance(tff.create_feature_extractor(args, torch.device("cpu")), FakeExtractor)
    args = tff.apply_model_preset(tff.build_parser().parse_args(["--model_preset", "uni"]))
    assert isinstance(tff.create_feature_extractor(args, torch.device("cpu")), FakeExtractor)
    args = tff.apply_model_preset(tff.build_parser().parse_args(["--model_preset", "derm_foundation"]))
    assert isinstance(tff.create_feature_extractor(args, torch.device("cpu")), FakeExtractor)
    args = tff.apply_model_preset(tff.build_parser().parse_args(["--model_preset", "chief", "--checkpoint_path", "x.pth"]))
    assert isinstance(tff.create_feature_extractor(args, torch.device("cpu")), FakeExtractor)
    args = tff.apply_model_preset(tff.build_parser().parse_args(["--model_preset", "medimageinsight"]))
    assert isinstance(tff.create_feature_extractor(args, torch.device("cpu")), FakeExtractor)

    assert len(calls) == 5


def test_dataset_samples_reads_subset_indices():
    base_dataset = types.SimpleNamespace(
        samples=[("image0.png", 0), ("image1.png", 1), ("image2.png", 0)]
    )
    subset = types.SimpleNamespace(dataset=base_dataset, indices=[2, 0])

    assert tff.dataset_samples(subset) == [("image2.png", 0), ("image0.png", 0)]
    assert tff.dataset_samples(base_dataset) == base_dataset.samples


def test_standardize_train_test_uses_training_statistics():
    train = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
    test = torch.tensor([[5.0, 10.0]])

    standardized_train, standardized_test = tff.standardize_train_test(train, test)

    assert torch.allclose(standardized_train.mean(dim=0), torch.zeros(2), atol=1e-6)
    assert standardized_test.tolist()[0][0] == pytest.approx((5.0 - 2.0) / torch.std(train[:, 0]).item())


def test_compute_class_weights_balances_inverse_frequency():
    labels = torch.tensor([0, 0, 0, 1])

    weights = tff.compute_class_weights(labels, num_classes=2)

    assert weights[0].item() == pytest.approx(4 / (2 * 3))
    assert weights[1].item() == pytest.approx(4 / (2 * 1))


def test_train_feature_classifier_learns_separable_linear_features():
    args = types.SimpleNamespace(
        standardize_features=True,
        classifier_batch_size=4,
        classifier_hidden_dim=0,
        dropout=0.0,
        class_weight="none",
        lr=0.1,
        weight_decay=0.0,
        num_epochs=25,
        log_interval=100,
    )
    features = tff.FeatureTensors(
        train_features=torch.tensor([[-3.0], [-2.0], [2.0], [3.0]]),
        train_labels=torch.tensor([0, 0, 1, 1]),
        test_features=torch.tensor([[-4.0], [4.0]]),
        test_labels=torch.tensor([0, 1]),
    )

    _, metrics = tff.train_feature_classifier(features, num_classes=2, args=args, device=torch.device("cpu"))

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)


def test_build_result_name_includes_feature_model_and_classifier():
    args = tff.build_parser().parse_args(
        [
            "--dataset",
            "breakhis",
            "--magnification",
            "100",
            "--feature_model",
            "torchvision:resnet50",
            "--feature_backend",
            "torchvision",
            "--classifier_hidden_dim",
            "256",
            "--class_weight",
            "balanced",
            "--num_epochs",
            "5",
        ]
    )

    assert tff.build_result_name(args) == (
        "foundation_breakhis_100X_0p3_torchvision-resnet50_mlp256_ep5_lr0p001_cwbalanced"
    )
