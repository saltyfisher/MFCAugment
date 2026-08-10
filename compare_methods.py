import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_METHODS = (
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

ABLATION_METHODS = (
    "mfc",
    "mfc_random_assign",
    "mfc_random_pool_group",
    "mfc_random_pool_random_assign",
    "mfc_representative_eval",
)

FOUNDATION_MODEL_PRESETS = {
    "dinov2": "facebook/dinov2-base",
    "biomedclip": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "derm_foundation": "google/derm-foundation",
    "path_foundation": "google/path-foundation",
    "chief": "CHIEF_CTransPath",
    "uni": "hf-hub:MahmoodLab/UNI",
    "virchow": "hf-hub:paige-ai/Virchow",
    "medimageinsight": "lion-ai/MedImageInsights",
}


@dataclass(frozen=True)
class SourceSpec:
    name: str
    local_path: str
    git_url: str


@dataclass(frozen=True)
class MethodSpec:
    key: str
    description: str
    runner: str
    strategy: str = ""
    mfc_args: tuple[str, ...] = ()
    source_specs: tuple[SourceSpec, ...] = ()
    required_paths: tuple[str, ...] = ()
    dataset_filter: tuple[str, ...] = ()


FASTAA_SOURCE = SourceSpec(
    name="Fast AutoAugment",
    local_path="fastaa",
    git_url="https://github.com/kakaobrain/fast-autoaugment.git",
)
CHIEF_SOURCE = SourceSpec(
    name="CHIEF",
    local_path="CHIEF",
    git_url="https://github.com/hms-dbmi/CHIEF.git",
)
ADAAUGMENT_SOURCE = SourceSpec(
    name="AdaAugment",
    local_path="external_methods/AdaAugment",
    git_url="https://github.com/Jackbrocp/AdaAugment.git",
)
ENTAUGMENT_SOURCE = SourceSpec(
    name="EntAugment",
    local_path="external_methods/EntAugment",
    git_url="https://github.com/Jackbrocp/EntAugment.git",
)
FREEAUGMENT_SOURCE = SourceSpec(
    name="FreeAugment",
    local_path="external_methods/FreeAugment",
    git_url="https://github.com/TomBekor/FreeAugment.git",
)
SRA_SOURCE = SourceSpec(
    name="Sample-aware RandAugment",
    local_path="external_methods/Sample-awareRandAugment",
    git_url="https://github.com/ainieli/Sample-awareRandAugment.git",
)


METHODS = {
    "baseline": MethodSpec(
        key="baseline",
        description="No extra augmentation",
        runner="train_mfc",
    ),
    "randaugment": MethodSpec(
        key="randaugment",
        description="Torchvision RandAugment",
        runner="train_mfc",
        strategy="randaugment",
    ),
    "trivialaugment": MethodSpec(
        key="trivialaugment",
        description="Torchvision TrivialAugmentWide",
        runner="train_mfc",
        strategy="trivialaugment",
    ),
    "randaugment_raw": MethodSpec(
        key="randaugment_raw",
        description="Local RandAugment implementation",
        runner="train_mfc",
        strategy="randaugment_raw",
    ),
    "trivialaugment_raw": MethodSpec(
        key="trivialaugment_raw",
        description="Local TrivialAugmentWide implementation",
        runner="train_mfc",
        strategy="trivialaugment_raw",
    ),
    "mfc": MethodSpec(
        key="mfc",
        description="Full HSDO-AA/MFC: searched policy pool with difficulty-matched assignment",
        runner="train_mfc",
        mfc_args=("--mfc", "--online", "--bayes", "--group", "--diff_c"),
    ),
    "mfc_random_assign": MethodSpec(
        key="mfc_random_assign",
        description="Searched policy pool with random sample-to-policy assignment",
        runner="train_mfc",
        mfc_args=("--mfc", "--online", "--bayes", "--diff_c"),
    ),
    "mfc_random_pool_group": MethodSpec(
        key="mfc_random_pool_group",
        description="Uniform random policy pool with difficulty-matched assignment",
        runner="train_mfc",
        mfc_args=("--mfc", "--online", "--bayes", "--group", "--diff_c", "--policy_pool_source", "random"),
    ),
    "mfc_random_pool_random_assign": MethodSpec(
        key="mfc_random_pool_random_assign",
        description="Uniform random policy pool with random sample-to-policy assignment",
        runner="train_mfc",
        mfc_args=("--mfc", "--online", "--bayes", "--diff_c", "--policy_pool_source", "random"),
    ),
    "mfc_representative_eval": MethodSpec(
        key="mfc_representative_eval",
        description="MFC diagnostic with representative eval subsampling",
        runner="train_mfc",
        mfc_args=(
            "--mfc",
            "--online",
            "--bayes",
            "--group",
            "--diff_c",
            "--mfc_eval_sampling",
            "representative",
        ),
    ),
    "fastaa": MethodSpec(
        key="fastaa",
        description="Fast AutoAugment with a project-local unified-search-space adapter",
        runner="external_adapter",
        source_specs=(FASTAA_SOURCE,),
        required_paths=("external_methods/adapters/train_fastaa.py",),
    ),
    "adaaugment": MethodSpec(
        key="adaaugment",
        description="AdaAugment with a project-local unified-search-space adapter",
        runner="external_adapter",
        source_specs=(ADAAUGMENT_SOURCE,),
        required_paths=("external_methods/adapters/train_adaaugment.py",),
    ),
    "entaugment": MethodSpec(
        key="entaugment",
        description="EntAugment with a project-local unified-search-space adapter",
        runner="external_adapter",
        source_specs=(ENTAUGMENT_SOURCE,),
        required_paths=("external_methods/adapters/train_entaugment.py",),
    ),
    "freeaugment": MethodSpec(
        key="freeaugment",
        description="FreeAugment with a project-local unified-search-space adapter",
        runner="external_adapter",
        source_specs=(FREEAUGMENT_SOURCE,),
        required_paths=("external_methods/adapters/train_freeaugment.py",),
    ),
    "sample_aware_randaugment": MethodSpec(
        key="sample_aware_randaugment",
        description="Sample-aware RandAugment with a project-local unified-search-space adapter",
        runner="external_adapter",
        source_specs=(SRA_SOURCE,),
        required_paths=("external_methods/adapters/train_sample_aware_randaugment.py",),
    ),
    "foundation_features": MethodSpec(
        key="foundation_features",
        description="Frozen vision foundation-model features with a lightweight classifier",
        runner="foundation_features",
        required_paths=("train_foundation_features.py",),
    ),
    "chief": MethodSpec(
        key="chief",
        description="CHIEF/CTransPath feature extractor baseline",
        runner="train_chief",
        source_specs=(CHIEF_SOURCE,),
        required_paths=("./CHIEF/model_weight/CHIEF_CTransPath.pth",),
        dataset_filter=("breakhis",),
    ),
}


@dataclass
class Experiment:
    method: str
    dataset: str
    magnification: str | None
    command: list[str]
    result_path: Path
    missing_sources: list[SourceSpec] = field(default_factory=list)
    missing_required_paths: list[Path] = field(default_factory=list)


def format_name_value(value):
    if isinstance(value, float):
        value = f"{value:g}"
    value = str(value)
    return value.replace("-", "m").replace(".", "p").replace("/", "-").replace("\\", "-").replace(":", "-")


def add_if_non_default(parts, name, value, default, prefix=None):
    if value != default:
        parts.append(f"{prefix or name}{format_name_value(value)}")


def normalize_methods(methods):
    if not methods or methods == ["default"]:
        return list(DEFAULT_METHODS)
    if methods == ["ablation"]:
        return list(ABLATION_METHODS)
    if methods == ["all"]:
        return list(METHODS)
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        raise ValueError(f"Unknown method(s): {', '.join(unknown)}")
    return methods


def expand_dataset_pairs(datasets, magnifications):
    pairs = []
    for dataset in datasets:
        if dataset == "breakhis":
            for magnification in magnifications:
                pairs.append((dataset, magnification))
        else:
            pairs.append((dataset, None))
    return pairs


def source_status(method, root):
    missing = []
    for source in method.source_specs:
        if not (root / source.local_path).exists():
            missing.append(source)
    return missing


def required_path_status(method, root):
    return [Path(path) for path in method.required_paths if not (root / path).exists()]


def build_train_mfc_command(args, method, dataset, magnification):
    command = [
        args.python,
        "train_mfc.py",
        "--data_dir",
        args.data_dir,
        "--dataset",
        dataset,
        "--model",
        args.model,
        "--batch_size",
        str(args.batch_size),
        "--num_epochs",
        str(args.num_epochs),
        "--lr",
        str(args.lr),
        "--dropout_rate",
        str(args.dropout_rate),
        "--optimizer",
        args.optimizer,
        "--weight_decay",
        str(args.weight_decay),
        "--test_split",
        str(args.test_split),
        "--num_trials",
        str(args.num_trials),
        "--bayes_max_eval",
        str(args.bayes_max_eval),
        "--bayes_topk",
        str(args.bayes_topk),
        "--bayes_rep",
        str(args.bayes_rep),
        "--mfc_eval_sample_ratio",
        str(args.mfc_eval_sample_ratio),
        "--mfc_eval_sampling",
        args.mfc_eval_sampling,
        "--mfc_eval_sample_seed",
        str(args.mfc_eval_sample_seed),
        "--mfc_eval_metric",
        args.mfc_eval_metric,
        "--mfc_refresh_interval",
        str(args.mfc_refresh_interval),
        "--uncertainty",
        args.uncertainty,
        "--subset_sigma",
        str(args.subset_sigma),
        "--l",
        str(args.l),
        "--num_ops",
        str(args.num_ops),
        "--mag_bin",
        str(args.mag_bin),
        "--prob_bin",
        str(args.prob_bin),
    ]
    if magnification is not None:
        command.extend(["--magnification", magnification])
    if method.strategy:
        command.extend(["--strategy", method.strategy])
    command.extend(method.mfc_args)
    if args.gpu:
        command.extend(["--gpu", "--device", str(args.device)])
    if args.pretrain:
        command.append("--pretrain")
    if args.use_prob:
        command.append("--use_prob")
    if args.testing:
        command.append("--testing")
    if not args.resize:
        command.append("--resize")
    return command


def build_train_fastaa_command(args, dataset, magnification):
    command = [
        args.python,
        "train_fastaa.py",
        "--data_dir",
        args.data_dir,
        "--dataset",
        dataset,
        "--model",
        args.model,
        "--batch_size",
        str(args.batch_size),
        "--num_epochs",
        str(args.num_epochs),
        "--lr",
        str(args.lr),
        "--test_split",
        str(args.test_split),
        "--num_trials",
        str(args.num_trials),
        "--device",
        str(args.device),
    ]
    if magnification is not None:
        command.extend(["--magnification", magnification])
    return command


def build_train_chief_command(args, dataset, magnification):
    data_root = Path(args.data_dir)
    if dataset == "breakhis" and magnification is not None:
        direct = data_root / magnification
        data_root = direct if direct.exists() else data_root

    command = [
        args.python,
        "train_chief.py",
        "--data_root",
        str(data_root),
        "--dataset",
        dataset,
        "--magnification",
        magnification or "40",
        "--model_path",
        args.chief_model_path,
        "--num_epochs",
        str(args.num_epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--test_split",
        str(args.test_split),
        "--num_trials",
        str(args.num_trials),
        "--device",
        "cuda" if args.gpu else "cpu",
    ]
    return command


def build_foundation_features_command(args, dataset, magnification):
    command = [
        args.python,
        "train_foundation_features.py",
        "--data_dir",
        args.data_dir,
        "--dataset",
        dataset,
        "--test_split",
        str(args.test_split),
        "--num_trials",
        str(args.num_trials),
    ]
    if args.model_preset:
        command.extend(["--model_preset", args.model_preset])
    command.extend([
        "--feature_backend",
        args.feature_backend,
        "--feature_model",
        args.feature_model,
        "--feature_batch_size",
        str(args.feature_batch_size),
        "--classifier_batch_size",
        str(args.classifier_batch_size),
        "--num_epochs",
        str(args.num_epochs),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--classifier_hidden_dim",
        str(args.classifier_hidden_dim),
        "--dropout",
        str(args.dropout_rate),
        "--class_weight",
        args.class_weight,
    ])
    if magnification is not None:
        command.extend(["--magnification", magnification])
    if args.checkpoint_path:
        command.extend(["--checkpoint_path", args.checkpoint_path])
    if args.timm_kwargs:
        command.extend(["--timm_kwargs", args.timm_kwargs])
    if args.keras_output_key != "embedding":
        command.extend(["--keras_output_key", args.keras_output_key])
    if args.keras_image_size is not None:
        command.extend(["--keras_image_size", str(args.keras_image_size)])
    if args.medimageinsight_model_dir != "2024.09.27":
        command.extend(["--medimageinsight_model_dir", args.medimageinsight_model_dir])
    if args.medimageinsight_vision_model_name != "medimageinsigt-v1.0.0.pt":
        command.extend(["--medimageinsight_vision_model_name", args.medimageinsight_vision_model_name])
    if args.medimageinsight_language_model_name != "language_model.pth":
        command.extend(["--medimageinsight_language_model_name", args.medimageinsight_language_model_name])
    if args.gpu:
        command.extend(["--gpu", "--device", str(args.device)])
    if args.no_feature_cache:
        command.append("--no_feature_cache")
    if args.no_standardize_features:
        command.append("--no_standardize_features")
    if args.trust_remote_code:
        command.append("--trust_remote_code")
    return command


def build_external_adapter_command(args, method, dataset, magnification):
    adapter_path = Path("external_methods") / "adapters" / f"train_{method.key}.py"
    command = [
        args.python,
        str(adapter_path),
        "--project_root",
        args.project_root,
        "--source_root",
        str(Path(args.project_root) / method.source_specs[0].local_path),
        "--data_dir",
        args.data_dir,
        "--dataset",
        dataset,
        "--model",
        args.model,
        "--batch_size",
        str(args.batch_size),
        "--num_epochs",
        str(args.num_epochs),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--test_split",
        str(args.test_split),
        "--num_trials",
        str(args.num_trials),
        "--search_space",
        "project",
        "--search_space_module",
        args.search_space_module,
        "--num_ops",
        str(args.num_ops),
        "--mag_bin",
        str(args.mag_bin),
        "--prob_bin",
        str(args.prob_bin),
    ]
    if magnification is not None:
        command.extend(["--magnification", magnification])
    if args.gpu:
        command.extend(["--gpu", "--device", str(args.device)])
    if args.testing:
        command.append("--testing")
    return command


def train_mfc_result_name(args, method, dataset, magnification):
    parts = [format_name_value(dataset), format_name_value(args.test_split)]
    if dataset == "breakhis" and magnification is not None:
        parts.append(f"{format_name_value(magnification)}X")
    if method.mfc_args:
        parts.append("mfc")
    elif method.strategy:
        parts.append(format_name_value(method.strategy))
    parts.append(format_name_value(args.model))

    if "--online" in method.mfc_args:
        parts.append("online")
    if args.testing:
        parts.append("testing")
    if args.pretrain:
        parts.append("pretrain")
    if not args.resize:
        parts.append("noresize")

    if method.mfc_args:
        add_if_non_default(parts, "num_ops", args.num_ops, 2, "ops")
        if args.use_prob:
            parts.append("useprob")
        add_if_non_default(parts, "mag_bin", args.mag_bin, 31, "magbin")
        add_if_non_default(parts, "prob_bin", args.prob_bin, 10, "probbin")
        add_if_non_default(parts, "l", args.l, 1, "l")
        parts.append("bayes")
        parts.extend(
            [
                f"eval{format_name_value(args.bayes_max_eval)}",
                f"topk{format_name_value(args.bayes_topk)}",
                f"rep{format_name_value(args.bayes_rep)}",
                f"ratio{format_name_value(args.mfc_eval_sample_ratio)}",
            ]
        )
        add_if_non_default(parts, "mfc_eval_metric", args.mfc_eval_metric, "kl", "metric")
        add_if_non_default(parts, "mfc_eval_sampling", args.mfc_eval_sampling, "uniform", "sampling")
        add_if_non_default(parts, "mfc_eval_sample_seed", getattr(args, "mfc_eval_sample_seed", 0), 0, "evalseed")
        if "--policy_pool_source" in method.mfc_args:
            parts.append("poolrandom")
            add_if_non_default(parts, "policy_pool_seed", args.policy_pool_seed, 0, "poolseed")
        add_if_non_default(parts, "mfc_refresh_interval", args.mfc_refresh_interval, 40, "refresh")
        add_if_non_default(parts, "uncertainty", args.uncertainty, "entropy", "unc")
        add_if_non_default(parts, "subset_sigma", args.subset_sigma, 0.15, "sigma")
        if "--group" in method.mfc_args:
            parts.append("group")
        if "--diff_c" in method.mfc_args:
            parts.append("diffc")

    add_if_non_default(parts, "batch_size", args.batch_size, 32, "bs")
    add_if_non_default(parts, "num_epochs", args.num_epochs, 180, "ep")
    add_if_non_default(parts, "lr", args.lr, 0.001, "lr")
    add_if_non_default(parts, "dropout_rate", args.dropout_rate, 0.5, "drop")
    if args.optimizer != "adam":
        parts.append(format_name_value(args.optimizer))
    add_if_non_default(parts, "weight_decay", args.weight_decay, 1e-4, "wd")
    return "_".join(parts)


def result_path_for(args, method, dataset, magnification):
    if method.runner == "train_mfc":
        return Path("result") / f"{train_mfc_result_name(args, method, dataset, magnification)}.csv"
    if method.runner == "train_fastaa":
        parts = ["fastaa", dataset]
        if dataset == "breakhis" and magnification is not None:
            parts.append(f"{magnification}X")
        return Path("result") / ("_".join(parts) + ".csv")
    if method.runner == "train_chief":
        parts = ["chief", dataset]
        if magnification is not None:
            parts.append(magnification)
        return Path("result") / ("_".join(parts) + ".csv")
    if method.runner == "foundation_features":
        feature_model = FOUNDATION_MODEL_PRESETS.get(args.model_preset, args.feature_model)
        parts = [
            "foundation",
            dataset,
            format_name_value(args.test_split),
            format_name_value(feature_model),
        ]
        if dataset == "breakhis" and magnification is not None:
            parts.insert(2, f"{format_name_value(magnification)}X")
        if args.classifier_hidden_dim:
            parts.append(f"mlp{format_name_value(args.classifier_hidden_dim)}")
        else:
            parts.append("linear")
        parts.extend([f"ep{args.num_epochs}", f"lr{format_name_value(args.lr)}"])
        if args.class_weight != "none":
            parts.append(f"cw{args.class_weight}")
        return Path("result") / ("_".join(parts) + ".csv")
    if method.runner == "external_adapter":
        parts = [method.key, dataset, format_name_value(args.test_split), format_name_value(args.model)]
        if dataset == "breakhis" and magnification is not None:
            parts.insert(2, f"{format_name_value(magnification)}X")
        add_if_non_default(parts, "num_ops", args.num_ops, 2, "ops")
        add_if_non_default(parts, "mag_bin", args.mag_bin, 31, "magbin")
        add_if_non_default(parts, "prob_bin", args.prob_bin, 10, "probbin")
        add_if_non_default(parts, "batch_size", args.batch_size, 32, "bs")
        add_if_non_default(parts, "num_epochs", args.num_epochs, 180, "ep")
        return Path("result") / ("_".join(parts) + ".csv")
    raise ValueError(f"Unsupported runner: {method.runner}")


def build_command(args, method, dataset, magnification):
    if method.runner == "train_mfc":
        return build_train_mfc_command(args, method, dataset, magnification)
    if method.runner == "train_fastaa":
        return build_train_fastaa_command(args, dataset, magnification)
    if method.runner == "train_chief":
        return build_train_chief_command(args, dataset, magnification)
    if method.runner == "foundation_features":
        return build_foundation_features_command(args, dataset, magnification)
    if method.runner == "external_adapter":
        return build_external_adapter_command(args, method, dataset, magnification)
    raise ValueError(f"Unsupported runner: {method.runner}")


def build_experiments(args):
    root = Path(args.project_root)
    experiments = []
    for method_key in normalize_methods(args.methods):
        method = METHODS[method_key]
        for dataset, magnification in expand_dataset_pairs(args.datasets, args.magnifications):
            if method.dataset_filter and dataset not in method.dataset_filter:
                continue
            missing = source_status(method, root)
            missing_required_paths = required_path_status(method, root)
            experiments.append(
                Experiment(
                    method=method_key,
                    dataset=dataset,
                    magnification=magnification,
                    command=build_command(args, method, dataset, magnification),
                    result_path=result_path_for(args, method, dataset, magnification),
                    missing_sources=missing,
                    missing_required_paths=missing_required_paths,
                )
            )
    return experiments


def command_to_text(command):
    return subprocess.list2cmdline(command)


def download_missing_sources(experiments, root):
    seen = set()
    for experiment in experiments:
        for source in experiment.missing_sources:
            if source.local_path in seen:
                continue
            seen.add(source.local_path)
            target = root / source.local_path
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", source.git_url, str(target)], cwd=root, check=True)


def read_metric_csv(path):
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if not row.get("metric"):
                continue
            rows.append(row)
    return rows


def write_plan_csv(path, experiments):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "method",
                "dataset",
                "magnification",
                "result_path",
                "missing_sources",
                "missing_required_paths",
                "command",
            ],
        )
        writer.writeheader()
        for experiment in experiments:
            writer.writerow(
                {
                    "method": experiment.method,
                    "dataset": experiment.dataset,
                    "magnification": experiment.magnification or "",
                    "result_path": str(experiment.result_path),
                    "missing_sources": ";".join(source.name for source in experiment.missing_sources),
                    "missing_required_paths": ";".join(str(path) for path in experiment.missing_required_paths),
                    "command": command_to_text(experiment.command),
                }
            )


def resolve_result_path(result_path, root):
    if result_path.is_absolute():
        return result_path
    return root / result_path


def write_summary_csv(path, experiments, root=None):
    root = path.parent if root is None else Path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "method",
                "dataset",
                "magnification",
                "metric",
                "mean",
                "std",
                "max",
                "min",
                "result_path",
            ],
        )
        writer.writeheader()
        for experiment in experiments:
            metric_path = resolve_result_path(experiment.result_path, root)
            for row in read_metric_csv(metric_path):
                writer.writerow(
                    {
                        "method": experiment.method,
                        "dataset": experiment.dataset,
                        "magnification": experiment.magnification or "",
                        "metric": row.get("metric", ""),
                        "mean": row.get("mean", ""),
                        "std": row.get("std", ""),
                        "max": row.get("max", ""),
                        "min": row.get("min", ""),
                        "result_path": str(metric_path),
                    }
                )


def run_experiments(args, experiments):
    for experiment in experiments:
        if experiment.missing_sources:
            names = ", ".join(source.name for source in experiment.missing_sources)
            raise FileNotFoundError(
                f"{experiment.method} requires missing source(s): {names}. "
                "Run with --download_sources first or remove that method."
            )
        if experiment.missing_required_paths:
            paths = ", ".join(str(path) for path in experiment.missing_required_paths)
            raise FileNotFoundError(
                f"{experiment.method} requires missing file(s): {paths}. "
                "Provide the files or remove that method."
            )
        if args.skip_existing and experiment.result_path.exists():
            print(f"SKIP {experiment.method} {experiment.dataset}: {experiment.result_path} exists")
            continue
        print(command_to_text(experiment.command))
        subprocess.run(experiment.command, cwd=args.project_root, check=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Run and summarize method comparison experiments.")
    parser.add_argument("--project_root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--methods", nargs="+", default=["default"], help="Method keys, default, or all")
    parser.add_argument("--datasets", nargs="+", default=["chestct"])
    parser.add_argument("--magnifications", nargs="+", default=["40"], help="BreakHis magnifications")
    parser.add_argument("--data_dir", default=".")
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=180)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--test_split", type=float, default=0.2)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--resize", action="store_true", default=True)
    parser.add_argument("--no_resize", dest="resize", action="store_false")
    parser.add_argument("--num_ops", type=int, default=2)
    parser.add_argument("--use_prob", action="store_true")
    parser.add_argument("--mag_bin", type=int, default=31)
    parser.add_argument("--prob_bin", type=int, default=10)
    parser.add_argument("--bayes_max_eval", type=int, default=100)
    parser.add_argument("--bayes_topk", type=float, default=0.1)
    parser.add_argument("--bayes_rep", type=int, default=2)
    parser.add_argument("--policy_pool_seed", type=int, default=0)
    parser.add_argument("--mfc_eval_sample_ratio", type=float, default=0.2)
    parser.add_argument("--mfc_eval_sampling", default="uniform", choices=["uniform", "representative"])
    parser.add_argument("--mfc_eval_sample_seed", type=int, default=0)
    parser.add_argument("--mfc_eval_metric", default="kl", choices=["kl", "mmd"])
    parser.add_argument("--search_space_module", default="core.augmentations")
    parser.add_argument("--mfc_refresh_interval", type=int, default=40)
    parser.add_argument("--uncertainty", default="entropy", choices=["entropy", "nll", "product"])
    parser.add_argument("--subset_sigma", type=float, default=0.15)
    parser.add_argument("--l", type=int, default=1)
    parser.add_argument("--chief_model_path", default="./CHIEF/model_weight/CHIEF_CTransPath.pth")
    parser.add_argument("--model_preset", default="", choices=[""] + sorted(FOUNDATION_MODEL_PRESETS))
    parser.add_argument(
        "--feature_backend",
        default="auto",
        choices=[
            "auto",
            "hf",
            "torchvision",
            "torchvision_checkpoint",
            "openclip",
            "timm",
            "keras",
            "chief",
            "medimageinsight",
        ],
    )
    parser.add_argument("--feature_model", default="facebook/dinov2-base")
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--timm_kwargs", default="")
    parser.add_argument("--keras_output_key", default="embedding")
    parser.add_argument("--keras_image_size", type=int, default=None)
    parser.add_argument("--medimageinsight_model_dir", default="2024.09.27")
    parser.add_argument("--medimageinsight_vision_model_name", default="medimageinsigt-v1.0.0.pt")
    parser.add_argument("--medimageinsight_language_model_name", default="language_model.pth")
    parser.add_argument("--feature_batch_size", type=int, default=16)
    parser.add_argument("--classifier_batch_size", type=int, default=128)
    parser.add_argument("--classifier_hidden_dim", type=int, default=0)
    parser.add_argument("--class_weight", default="none", choices=["none", "balanced"])
    parser.add_argument("--no_feature_cache", action="store_true")
    parser.add_argument("--no_standardize_features", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--download_sources", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan_csv", default="result/method_comparison_plan.csv")
    parser.add_argument("--summary_csv", default="result/method_comparison_summary.csv")
    parser.add_argument("--summary_only", action="store_true")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        experiments = build_experiments(args)
    except ValueError as exc:
        parser.error(str(exc))

    root = Path(args.project_root)
    if args.download_sources:
        download_missing_sources(experiments, root)
        experiments = build_experiments(args)

    write_plan_csv(root / args.plan_csv, experiments)
    write_summary_csv(root / args.summary_csv, experiments, root)

    if args.summary_only:
        print(f"Wrote summary: {root / args.summary_csv}")
        return 0

    for experiment in experiments:
        if experiment.missing_sources:
            marker = "MISSING_SOURCE"
        elif experiment.missing_required_paths:
            marker = "MISSING_FILE"
        else:
            marker = "READY"
        print(
            f"[{marker}] {experiment.method} | {experiment.dataset} | "
            f"{experiment.magnification or '-'} | {command_to_text(experiment.command)}"
        )

    print(f"Wrote plan: {root / args.plan_csv}")
    if args.execute:
        run_experiments(args, experiments)
        write_summary_csv(root / args.summary_csv, experiments, root)
        print(f"Wrote summary: {root / args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
