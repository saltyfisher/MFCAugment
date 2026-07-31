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
    assert args.mfc_eval_sampling == "representative"
    assert args.mfc_eval_sample_seed == 0
    assert args.bayes_topk == pytest.approx(0.1)
    assert args.policy_pool_source == "search"
    assert args.policy_pool_seed == 0
    assert args.mfc_refresh_interval == 40
    assert args.mfc_eval_metric == "kl"
    assert args.uncertainty == "entropy"
    assert args.subset_sigma == pytest.approx(0.15)


def test_parser_accepts_mfc_eval_sample_ratio(train_mfc):
    parser = train_mfc.build_parser()
    args = parser.parse_args(["--mfc_eval_sample_ratio", "0.25"])

    assert args.mfc_eval_sample_ratio == pytest.approx(0.25)


def test_parser_accepts_mfc_eval_sampling_mode_and_seed(train_mfc):
    parser = train_mfc.build_parser()
    args = parser.parse_args(["--mfc_eval_sampling", "uniform", "--mfc_eval_sample_seed", "13"])

    assert args.mfc_eval_sampling == "uniform"
    assert args.mfc_eval_sample_seed == 13


def test_parser_accepts_mfc_eval_metric(train_mfc):
    parser = train_mfc.build_parser()
    args = parser.parse_args(["--mfc_eval_metric", "mmd"])

    assert args.mfc_eval_metric == "mmd"


def test_parser_rejects_invalid_mfc_eval_metric(train_mfc):
    parser = train_mfc.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--mfc_eval_metric", "wasserstein"])


def test_parser_rejects_invalid_mfc_eval_sampling_mode(train_mfc):
    parser = train_mfc.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--mfc_eval_sampling", "stratified"])


def test_parser_accepts_uncertainty_and_subset_sigma(train_mfc):
    parser = train_mfc.build_parser()
    args = parser.parse_args(["--uncertainty", "product", "--subset_sigma", "0.2"])

    assert args.uncertainty == "product"
    assert args.subset_sigma == pytest.approx(0.2)


def test_parser_rejects_invalid_uncertainty_and_subset_sigma(train_mfc):
    parser = train_mfc.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--uncertainty", "unknown"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--subset_sigma", "0"])


def test_parser_accepts_bayes_topk_ratio(train_mfc):
    parser = train_mfc.build_parser()
    args = parser.parse_args(["--bayes_topk", "0.25"])

    assert args.bayes_topk == pytest.approx(0.25)


def test_parser_accepts_random_policy_pool_source(train_mfc):
    parser = train_mfc.build_parser()
    args = parser.parse_args(["--policy_pool_source", "random", "--policy_pool_seed", "17"])

    assert args.policy_pool_source == "random"
    assert args.policy_pool_seed == 17


def test_parser_accepts_mfc_refresh_interval(train_mfc):
    parser = train_mfc.build_parser()
    args = parser.parse_args(["--mfc_refresh_interval", "20"])

    assert args.mfc_refresh_interval == 20


@pytest.mark.parametrize("value", ["0", "-1"])
def test_parser_rejects_invalid_mfc_refresh_interval(train_mfc, value):
    parser = train_mfc.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--mfc_refresh_interval", value])


def test_parser_rejects_invalid_policy_pool_source(train_mfc):
    parser = train_mfc.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--policy_pool_source", "tpe"])


@pytest.mark.parametrize("ratio", ["0", "1.2", "-0.1"])
def test_parser_rejects_invalid_bayes_topk_ratio(train_mfc, ratio):
    parser = train_mfc.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--bayes_topk", ratio])


def test_resolve_bayes_topk_count_uses_ratio_of_max_evals(train_mfc):
    assert train_mfc.resolve_bayes_topk_count(0.25, 40) == 10
    assert train_mfc.resolve_bayes_topk_count(0.01, 3) == 1
    assert train_mfc.resolve_bayes_topk_count(1.0, 7) == 7


def test_parser_controls_full_group_reevaluation(train_mfc):
    parser = train_mfc.build_parser()
    default_args = parser.parse_args([])
    explicit_false_args = parser.parse_args(["--reevaluate_full_groups", "false"])
    flag_false_args = parser.parse_args(["--no_reevaluate_full_groups"])

    assert default_args.reevaluate_full_groups is True
    assert explicit_false_args.reevaluate_full_groups is False
    assert flag_false_args.reevaluate_full_groups is False


def test_parser_accepts_single_parameter_sensitivity_options(train_mfc):
    parser = train_mfc.build_parser()
    args = parser.parse_args(
        [
            "--param_test",
            "mfc_eval_sample_ratio",
            "--param_values",
            "0.1",
            "0.3",
            "0.5",
        ]
    )

    assert args.param_test == "mfc_eval_sample_ratio"
    assert args.param_values == ["0.1", "0.3", "0.5"]


@pytest.mark.parametrize("ratio", ["0", "1", "1.2", "-0.1"])
def test_parser_rejects_invalid_mfc_eval_sample_ratio(train_mfc, ratio):
    parser = train_mfc.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--mfc_eval_sample_ratio", ratio])


def test_parse_parameter_test_values_converts_supported_types(train_mfc):
    assert train_mfc.parse_parameter_test_values("mfc_eval_sample_ratio", ["0.1", "0.5"]) == [
        pytest.approx(0.1),
        pytest.approx(0.5),
    ]
    assert train_mfc.parse_parameter_test_values("mfc_eval_sampling", ["representative", "uniform"]) == [
        "representative",
        "uniform",
    ]
    assert train_mfc.parse_parameter_test_values("mfc_eval_metric", ["kl", "mmd"]) == ["kl", "mmd"]
    assert train_mfc.parse_parameter_test_values("policy_pool_source", ["search", "random"]) == [
        "search",
        "random",
    ]
    assert train_mfc.parse_parameter_test_values("group", ["false", "true"]) == [False, True]
    assert train_mfc.parse_parameter_test_values("diff_c", ["0", "1"]) == [False, True]
    assert train_mfc.parse_parameter_test_values("l", ["1", "3"]) == [1, 3]
    assert train_mfc.parse_parameter_test_values("mfc_refresh_interval", ["20", "40"]) == [20, 40]
    assert train_mfc.parse_parameter_test_values("uncertainty", ["entropy", "nll"]) == ["entropy", "nll"]
    assert train_mfc.parse_parameter_test_values("subset_sigma", ["0.1", "0.2"]) == [
        pytest.approx(0.1),
        pytest.approx(0.2),
    ]


def test_parse_parameter_test_values_uses_defaults_when_values_are_omitted(train_mfc):
    assert train_mfc.parse_parameter_test_values("group", None) == [False, True]


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


def test_should_refresh_policy_uses_configured_interval_for_online_mode(train_mfc):
    args = types.SimpleNamespace(online=True, testing=False, mfc_refresh_interval=20)

    assert train_mfc.should_refresh_policy(args, epoch=20, epoch_start=1) is True
    assert train_mfc.should_refresh_policy(args, epoch=40, epoch_start=1) is True
    assert train_mfc.should_refresh_policy(args, epoch=21, epoch_start=1) is False


def test_should_refresh_policy_keeps_offline_and_testing_start_epoch_behavior(train_mfc):
    offline_args = types.SimpleNamespace(online=False, testing=False, mfc_refresh_interval=20)
    testing_args = types.SimpleNamespace(online=True, testing=True, mfc_refresh_interval=20)

    assert train_mfc.should_refresh_policy(offline_args, epoch=1, epoch_start=1) is True
    assert train_mfc.should_refresh_policy(offline_args, epoch=20, epoch_start=1) is False
    assert train_mfc.should_refresh_policy(testing_args, epoch=1, epoch_start=1) is True
    assert train_mfc.should_refresh_policy(testing_args, epoch=20, epoch_start=1) is False


def test_resolve_device_uses_cpu_when_gpu_is_disabled(train_mfc):
    args = types.SimpleNamespace(gpu=False, device=0)

    assert train_mfc.resolve_device(args).type == "cpu"


def test_build_idx_to_group_rejects_overlapping_groups(train_mfc):
    with pytest.raises(ValueError):
        train_mfc.build_idx_to_group([[0, 1], [1, 2]], require_disjoint=True)

    assert train_mfc.build_idx_to_group([[0, 1], [1, 2]], require_disjoint=False)[1] == 1


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
    assert metrics["unaugmented_ratio"] == pytest.approx(0.0)


def test_run_epoch_keeps_unassigned_group_samples_unaugmented(train_mfc):
    torch = train_mfc.torch
    calls = []

    def make_policy(policy_id):
        def apply(image):
            calls.append(policy_id)
            return train_mfc.transforms.ToTensor()(image)
        return apply

    batches = [
        (torch.rand(2, 3, 4, 4), torch.tensor([0, 1]), torch.tensor([0, 1])),
    ]
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2))

    metrics = train_mfc.run_epoch(
        model,
        batches,
        torch.nn.CrossEntropyLoss(),
        optimizer=None,
        policy=[make_policy(0), make_policy(1)],
        groups={0: 1},
        args=types.SimpleNamespace(group=True),
    )

    assert calls == [1]
    assert metrics["loss"] >= 0
    assert metrics["unaugmented_ratio"] == pytest.approx(0.5)


def test_prepare_output_config_builds_parameterized_names_and_paths(train_mfc):
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
            "--bayes",
            "--group",
            "--diff_c",
            "--mfc_eval_sample_ratio",
            "0.35",
            "--bayes_max_eval",
            "12",
            "--bayes_topk",
            "0.25",
            "--bayes_rep",
            "1",
            "--testing",
        ]
    )

    output = train_mfc.prepare_output_config(args)

    assert output.save_name == (
        "breakhis_0p2_100X_mfc_resnet34_online_testing_bayes_eval12_topk0p25_rep1_ratio0p35"
        "_group_diffc_proxy_GD"
    )
    assert output.save_path == Path("params_save") / "mfc"
    assert output.log_path == Path("logs") / "mfc"
    assert args.save_name == output.save_name
    assert args.proxy_log_path == output.log_path / "proxy"
    assert args.GD_save_path == output.save_path / "GD"


def test_build_save_name_marks_disabled_full_group_reevaluation(train_mfc):
    parser = train_mfc.build_parser()
    args = parser.parse_args(["--mfc", "--bayes", "--reevaluate_full_groups", "false"])

    assert "nofullreeval" in train_mfc.build_save_name(args)


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


def test_run_parameter_sensitivity_varies_one_parameter_at_a_time(train_mfc, monkeypatch):
    parser = train_mfc.build_parser()
    args = parser.parse_args(
        [
            "--param_test",
            "group",
            "--param_values",
            "false",
            "true",
            "--diff_c",
            "--l",
            "3",
        ]
    )
    calls = []
    written = {}

    def fake_run_trials(variant_args):
        calls.append(
            {
                "group": variant_args.group,
                "diff_c": variant_args.diff_c,
                "l": variant_args.l,
                "same_object": variant_args is args,
            }
        )
        accuracy = 0.8 if variant_args.group else 0.6
        return [{"accuracy": accuracy, "f1": accuracy - 0.1}]

    def fake_write_csv(base_args, param_name, records):
        written["param_name"] = param_name
        written["values"] = [record["value"] for record in records]

    monkeypatch.setattr(train_mfc, "run_trials", fake_run_trials)
    monkeypatch.setattr(train_mfc, "write_parameter_sensitivity_csv", fake_write_csv)

    records = train_mfc.run_parameter_sensitivity(args)

    assert calls == [
        {"group": False, "diff_c": True, "l": 3, "same_object": False},
        {"group": True, "diff_c": True, "l": 3, "same_object": False},
    ]
    assert [record["value"] for record in records] == [False, True]
    assert records[0]["stats"]["accuracy"]["mean"] == "0.6000"
    assert records[1]["stats"]["accuracy"]["mean"] == "0.8000"
    assert written == {"param_name": "group", "values": [False, True]}


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


def test_update_best_metrics_keeps_best_accuracy_for_best_f1(train_mfc):
    current_best = {"accuracy": 0.75, "f1": 0.8}
    weaker_metrics = {"accuracy": 0.9, "f1": 0.7}
    stronger_metrics = {"accuracy": 0.82, "f1": 0.85}

    best_metrics, best_f1 = train_mfc.update_best_metrics(weaker_metrics, current_best, 0.8)
    assert best_metrics == current_best
    assert best_f1 == pytest.approx(0.8)

    best_metrics, best_f1 = train_mfc.update_best_metrics(stronger_metrics, best_metrics, best_f1)
    assert best_metrics == stronger_metrics
    assert best_f1 == pytest.approx(0.85)


def test_format_test_epoch_metrics_includes_best_test_accuracy(train_mfc):
    metrics = {"loss": 0.12345, "accuracy": 0.81234}
    best_metrics = {"accuracy": 0.87654}

    assert train_mfc.format_test_epoch_metrics(2, 5, metrics, best_metrics) == (
        "Epoch [2/5] - Test Loss: 0.1235, Test Acc: 0.8123, Best Test Acc: 0.8765"
    )


def test_build_stats_csv_path_uses_parameterized_save_name(train_mfc):
    parser = train_mfc.build_parser()

    mfc_args = parser.parse_args(
        [
            "--dataset",
            "breakhis",
            "--magnification",
            "40",
            "--mfc",
            "--bayes",
            "--group",
            "--diff_c",
            "--mfc_eval_sample_ratio",
            "0.5",
        ]
    )
    assert train_mfc.build_stats_csv_path(mfc_args) == Path(
        "result/breakhis_0p2_40X_mfc_resnet18_bayes_eval100_topk0p1_rep2_ratio0p5_group_diffc.csv"
    )

    strategy_args = parser.parse_args(["--dataset", "chestct", "--strategy", "randaugment"])
    assert train_mfc.build_stats_csv_path(strategy_args) == Path("result/chestct_0p2_randaugment_resnet18.csv")
