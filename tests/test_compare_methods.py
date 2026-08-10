import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import compare_methods as cmp


def parse_args(args):
    return cmp.build_parser().parse_args(args)


def test_default_plan_builds_expected_core_methods(tmp_path):
    args = parse_args(["--project_root", str(tmp_path), "--data_dir", "D:/data", "--datasets", "chestct"])

    experiments = cmp.build_experiments(args)

    assert [experiment.method for experiment in experiments] == list(cmp.DEFAULT_METHODS)
    assert all(experiment.dataset == "chestct" for experiment in experiments)
    assert all(experiment.magnification is None for experiment in experiments)


def test_default_methods_match_main_comparison_set():
    assert cmp.DEFAULT_METHODS == (
        "baseline",
        "randaugment",
        "trivialaugment",
        "mfc",
        "fastaa",
        "adaaugment",
        "entaugment",
        "freeaugment",
        "sample_aware_randaugment",
    )


def test_ablation_methods_are_separate_from_main_comparison(tmp_path):
    args = parse_args(["--project_root", str(tmp_path), "--methods", "ablation", "--datasets", "chestct"])

    experiments = cmp.build_experiments(args)

    assert [experiment.method for experiment in experiments] == list(cmp.ABLATION_METHODS)


def test_external_methods_record_source_and_adapter_requirements(tmp_path):
    args = parse_args(
        [
            "--project_root",
            str(tmp_path),
            "--methods",
            "fastaa",
            "adaaugment",
            "entaugment",
            "freeaugment",
            "sample_aware_randaugment",
            "--datasets",
            "chestct",
        ]
    )

    experiments = cmp.build_experiments(args)

    assert [experiment.missing_sources[0].name for experiment in experiments] == [
        "Fast AutoAugment",
        "AdaAugment",
        "EntAugment",
        "FreeAugment",
        "Sample-aware RandAugment",
    ]
    assert [experiment.missing_required_paths[0].as_posix() for experiment in experiments] == [
        "external_methods/adapters/train_fastaa.py",
        "external_methods/adapters/train_adaaugment.py",
        "external_methods/adapters/train_entaugment.py",
        "external_methods/adapters/train_freeaugment.py",
        "external_methods/adapters/train_sample_aware_randaugment.py",
    ]
    assert all("--search_space_module" in experiment.command for experiment in experiments)
    assert all("core.augmentations" in experiment.command for experiment in experiments)


def test_breakhis_expands_requested_magnifications(tmp_path):
    args = parse_args(
        [
            "--project_root",
            str(tmp_path),
            "--datasets",
            "breakhis",
            "--magnifications",
            "40",
            "100",
            "--methods",
            "baseline",
        ]
    )

    experiments = cmp.build_experiments(args)

    assert [(experiment.dataset, experiment.magnification) for experiment in experiments] == [
        ("breakhis", "40"),
        ("breakhis", "100"),
    ]


def test_mfc_2x2_commands_encode_search_and_assignment_factors(tmp_path):
    args = parse_args(
        [
            "--project_root",
            str(tmp_path),
            "--datasets",
            "chestct",
            "--methods",
            "mfc",
            "mfc_random_assign",
            "mfc_random_pool_group",
            "mfc_random_pool_random_assign",
        ]
    )

    commands = {experiment.method: experiment.command for experiment in cmp.build_experiments(args)}

    assert "--group" in commands["mfc"]
    assert "--policy_pool_source" not in commands["mfc"]
    assert "--group" not in commands["mfc_random_assign"]
    assert commands["mfc_random_pool_group"].count("--group") == 1
    assert "--policy_pool_source" in commands["mfc_random_pool_group"]
    assert "--group" not in commands["mfc_random_pool_random_assign"]
    assert "--policy_pool_source" in commands["mfc_random_pool_random_assign"]


def test_source_status_detects_external_source_dirs(tmp_path):
    args = parse_args(["--project_root", str(tmp_path), "--methods", "fastaa", "chief", "--datasets", "breakhis"])

    experiments = cmp.build_experiments(args)

    assert {source.name for exp in experiments for source in exp.missing_sources} == {
        "Fast AutoAugment",
        "CHIEF",
    }

    (tmp_path / "fastaa").mkdir()
    (tmp_path / "CHIEF").mkdir()
    experiments = cmp.build_experiments(args)

    assert all(not experiment.missing_sources for experiment in experiments)


def test_chief_is_limited_to_breakhis(tmp_path):
    (tmp_path / "CHIEF").mkdir()
    args = parse_args(
        [
            "--project_root",
            str(tmp_path),
            "--methods",
            "chief",
            "--datasets",
            "chestct",
            "breakhis",
            "--magnifications",
            "40",
        ]
    )

    experiments = cmp.build_experiments(args)

    assert [(experiment.dataset, experiment.magnification) for experiment in experiments] == [("breakhis", "40")]


def test_chief_reports_missing_default_weight_file(tmp_path):
    (tmp_path / "CHIEF").mkdir()
    args = parse_args(["--project_root", str(tmp_path), "--methods", "chief", "--datasets", "breakhis"])

    experiment = cmp.build_experiments(args)[0]

    assert experiment.missing_sources == []
    assert experiment.missing_required_paths == [Path("CHIEF/model_weight/CHIEF_CTransPath.pth")]


def test_foundation_features_builds_explicit_large_model_command(tmp_path):
    args = parse_args(
        [
            "--project_root",
            str(tmp_path),
            "--methods",
            "foundation_features",
            "--datasets",
            "pad-ufes-20",
            "--data_dir",
            "E:/Data/MedicalImage",
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
    (tmp_path / "train_foundation_features.py").write_text("", encoding="utf-8")

    experiment = cmp.build_experiments(args)[0]

    assert experiment.method == "foundation_features"
    assert experiment.missing_required_paths == []
    assert experiment.result_path == Path(
        "result/foundation_pad-ufes-20_0p2_torchvision-resnet50_mlp256_ep5_lr0p001_cwbalanced.csv"
    )
    assert experiment.command[:6] == [
        sys.executable,
        "train_foundation_features.py",
        "--data_dir",
        "E:/Data/MedicalImage",
        "--dataset",
        "pad-ufes-20",
    ]
    assert "--feature_model" in experiment.command
    assert "torchvision:resnet50" in experiment.command
    assert "--class_weight" in experiment.command
    assert "balanced" in experiment.command


def test_foundation_features_accepts_specialized_model_preset(tmp_path):
    args = parse_args(
        [
            "--project_root",
            str(tmp_path),
            "--methods",
            "foundation_features",
            "--datasets",
            "pad-ufes-20",
            "--model_preset",
            "biomedclip",
            "--num_epochs",
            "3",
        ]
    )
    (tmp_path / "train_foundation_features.py").write_text("", encoding="utf-8")

    experiment = cmp.build_experiments(args)[0]

    assert "--model_preset" in experiment.command
    assert "biomedclip" in experiment.command
    assert experiment.result_path == Path(
        "result/foundation_pad-ufes-20_0p2_hfmhub-microsoft-BiomedCLIPmPubMedBERT_256mvit_base_patch16_224_linear_ep3_lr0p001.csv"
    )


def test_train_mfc_result_names_match_existing_convention(tmp_path):
    args = parse_args(
        [
            "--project_root",
            str(tmp_path),
            "--datasets",
            "breakhis",
            "--magnifications",
            "100",
            "--methods",
            "mfc",
            "--model",
            "resnet34",
            "--testing",
            "--mfc_eval_sample_ratio",
            "0.35",
            "--bayes_max_eval",
            "12",
            "--bayes_topk",
            "0.25",
            "--bayes_rep",
            "1",
        ]
    )

    experiment = cmp.build_experiments(args)[0]

    assert experiment.result_path == Path(
        "result/breakhis_0p2_100X_mfc_resnet34_online_testing_bayes_eval12_topk0p25_rep1_ratio0p35_group_diffc.csv"
    )


def test_train_mfc_result_names_include_non_default_comparison_knobs(tmp_path):
    args = parse_args(
        [
            "--project_root",
            str(tmp_path),
            "--datasets",
            "chestct",
            "--methods",
            "mfc",
            "--mfc_eval_sampling",
            "representative",
            "--mfc_eval_sample_seed",
            "9",
            "--uncertainty",
            "nll",
            "--subset_sigma",
            "0.2",
        ]
    )

    experiment = cmp.build_experiments(args)[0]

    assert "samplingrepresentative" in str(experiment.result_path)
    assert "evalseed9" in str(experiment.result_path)
    assert "uncnll" in str(experiment.result_path)
    assert "sigma0p2" in str(experiment.result_path)


def test_summary_csv_collects_existing_metric_files(tmp_path):
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    metric_path = result_dir / "chestct_0p2_resnet18_ep1.csv"
    metric_path.write_text(
        "metric,mean,std,max,min\naccuracy,0.8,0.1,0.9,0.7\nf1,0.75,0.05,0.8,0.7\n",
        encoding="utf-8",
    )
    args = parse_args(
        [
            "--project_root",
            str(tmp_path),
            "--methods",
            "baseline",
            "--datasets",
            "chestct",
            "--num_epochs",
            "1",
        ]
    )
    experiments = cmp.build_experiments(args)

    cmp.write_summary_csv(tmp_path / "summary.csv", experiments)

    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as csvfile:
        rows = list(csv.DictReader(csvfile))

    assert [(row["method"], row["metric"], row["mean"]) for row in rows] == [
        ("baseline", "accuracy", "0.8"),
        ("baseline", "f1", "0.75"),
    ]
