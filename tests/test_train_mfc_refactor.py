import importlib
import logging
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@pytest.fixture()
def train_mfc(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    stubs = {
        "aug_lib": _module("aug_lib"),
        "models": _module("models"),
        "data": _module("data", get_data=lambda *args, **kwargs: None),
        "networks": _module(
            "networks",
            get_model=lambda *args, **kwargs: None,
            num_class=lambda dataset: 4,
        ),
        "utils": _module("utils", initialize_setting=lambda: None),
        "theconf": _module("theconf", Config=object),
        "lr_scheduler": _module(
            "lr_scheduler",
            adjust_learning_rate_resnet=lambda *args, **kwargs: None,
        ),
        "metrics": _module("metrics", Accumulator=object),
        "tensorboardX": _module("tensorboardX", SummaryWriter=object),
        "warmup_scheduler": _module("warmup_scheduler", GradualWarmupScheduler=object),
        "common": _module("common", get_logger=lambda name: logging.getLogger(name)),
    }
    core_module = _module("core")
    stubs["core"] = core_module
    stubs["core.MFCAugment"] = _module(
        "core.MFCAugment",
        MFCAugment=lambda *args, **kwargs: ([], [], []),
    )
    stubs["core.augmentations"] = _module(
        "core.augmentations",
        MyAugment=object,
        RandAugment=object,
        TrivialAugmentWide=object,
    )

    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)

    sys.modules.pop("train_mfc", None)
    return importlib.import_module("train_mfc")


def test_build_parser_preserves_mfc_defaults(train_mfc):
    parser = train_mfc.build_parser()
    args = parser.parse_args([])

    assert args.dataset == "chestct"
    assert args.model == "resnet18"
    assert args.batch_size == 32
    assert args.num_epochs == 180
    assert args.num_trials == 10
    assert args.num_ops == 2
    assert args.resize is True
    assert args.mfc is False
    assert args.mfc_eval_sample_ratio == pytest.approx(0.2)


def test_parser_accepts_mfc_eval_sample_ratio(train_mfc):
    parser = train_mfc.build_parser()
    args = parser.parse_args(["--mfc_eval_sample_ratio", "0.25"])

    assert args.mfc_eval_sample_ratio == pytest.approx(0.25)


@pytest.mark.parametrize("ratio", ["0", "1", "1.2", "-0.1"])
def test_parser_rejects_invalid_mfc_eval_sample_ratio(train_mfc, ratio):
    parser = train_mfc.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--mfc_eval_sample_ratio", ratio])


@pytest.mark.parametrize(
    ("dataset", "expected"),
    [
        ("lymphoma", 2),
        ("breakhis", 8),
        ("chestct", 4),
        ("unknown", 4),
    ],
)
def test_get_cluster_count_matches_existing_dataset_rules(train_mfc, dataset, expected):
    assert train_mfc.get_cluster_count(dataset) == expected


def test_resolve_device_uses_cpu_when_gpu_is_disabled(train_mfc):
    args = types.SimpleNamespace(gpu=False, device=0)

    assert train_mfc.resolve_device(args).type == "cpu"


def test_run_epoch_applies_group_policy_per_sample_and_resets_each_batch(train_mfc):
    torch = train_mfc.torch
    calls = []

    def make_policy(policy_id):
        def apply(image):
            calls.append(policy_id)
            return train_mfc.transforms.ToTensor()(image)
        return apply

    batches = [
        (torch.rand(2, 3, 4, 4), torch.tensor([0, 1]), torch.tensor([0, 1])),
        (torch.rand(2, 3, 4, 4), torch.tensor([1, 0]), torch.tensor([2, 3])),
    ]
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2))

    metrics = train_mfc.run_epoch(
        model,
        batches,
        torch.nn.CrossEntropyLoss(),
        optimizer=None,
        policy=[make_policy(0), make_policy(1)],
        groups={0: 0, 1: 1, 2: 0, 3: 1},
        args=types.SimpleNamespace(group=True),
    )

    assert calls == [0, 1, 0, 1]
    assert metrics["loss"] >= 0


def test_prepare_output_config_builds_existing_names_and_paths(train_mfc):
    parser = train_mfc.build_parser()
    args = parser.parse_args(
        [
            "--dataset",
            "breakhis",
            "--magnification",
            "100",
            "--mfc",
            "--online",
            "--proxy",
            "--GD",
            "--model",
            "resnet34",
        ]
    )

    output = train_mfc.prepare_output_config(args)

    assert output.save_name == "breakhis_0.2_100X_mfc_resnet34_online_proxy_GD"
    assert output.save_path == Path("params_save") / "mfc"
    assert output.log_path == Path("logs") / "mfc"
    assert args.save_name == output.save_name
    assert args.proxy_log_path == output.log_path / "proxy"
    assert args.GD_save_path == output.save_path / "GD"


def test_run_trial_uses_configured_data_dir(train_mfc, monkeypatch, tmp_path):
    captured = {}
    args = types.SimpleNamespace(num_trials=1, data_dir="E:/Data/MedicalImage")

    monkeypatch.setattr(train_mfc, "create_model", lambda args: ("model", 4))
    monkeypatch.setattr(train_mfc, "create_optimizer", lambda model, args: "optimizer")
    monkeypatch.setattr(
        train_mfc,
        "prepare_output_config",
        lambda args: train_mfc.OutputConfig(tmp_path, tmp_path, "smoke"),
    )

    def fake_train_val(model, optimizer, num_classes, args, itrs, dataroot, *rest):
        captured["dataroot"] = dataroot
        return model, {"accuracy": 1.0}

    monkeypatch.setattr(train_mfc, "train_val", fake_train_val)

    train_mfc.run_trial(args, 0)

    assert captured["dataroot"] == "E:/Data/MedicalImage"


def test_refresh_mfc_policy_uses_true_groups_for_group_assignment(train_mfc, monkeypatch):
    mfc_module = sys.modules["core.MFCAugment"]
    monkeypatch.setattr(
        mfc_module,
        "MFCAugment",
        lambda *args, **kwargs: (["policy"], [[0]], [[1, 2]]),
    )
    monkeypatch.setattr(train_mfc, "build_policy_transforms", lambda policies, size, args: policies)
    args = types.SimpleNamespace(dataset="chestct", num_ops=2, group=True)

    policies, idx_to_group, raw_policies = train_mfc.refresh_mfc_policy(
        model="model",
        resize_size=(224, 224),
        data_list=["a", "b", "c"],
        label_list=[0, 1, 1],
        args=args,
    )

    assert policies == ["policy"]
    assert raw_policies == ["policy"]
    assert idx_to_group == {1: 0, 2: 0}


def test_build_stats_rows_formats_metric_values(train_mfc):
    rows = train_mfc.build_stats_rows(
        [
            {"accuracy": 0.8, "f1": 0.7},
            {"accuracy": 0.6, "f1": 0.9},
        ]
    )

    assert rows == {
        "accuracy": {"mean": "0.7000", "std": "0.1000", "max": "0.8000", "min": "0.6000"},
        "f1": {"mean": "0.8000", "std": "0.1000", "max": "0.9000", "min": "0.7000"},
    }


def test_build_stats_csv_path_matches_existing_naming(train_mfc):
    parser = train_mfc.build_parser()

    mfc_args = parser.parse_args(["--dataset", "breakhis", "--magnification", "40", "--mfc"])
    assert train_mfc.build_stats_csv_path(mfc_args) == Path("result/breakhis_40X_mfc_resnet18.csv")

    strategy_args = parser.parse_args(["--dataset", "chestct", "--strategy", "randaugment"])
    assert train_mfc.build_stats_csv_path(strategy_args) == Path("result/chestct_randaugment_resnet18.csv")
