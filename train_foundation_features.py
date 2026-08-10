import argparse
import base64
import csv
import io
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset, TensorDataset

from data import get_data, load_default_image_loader
from networks import num_class


SUPPORTED_DATASETS = ("chestct", "breakhis", "corona", "cifar-fs", "miniimagenet", "pad-ufes-20")
MODEL_PRESETS = {
    "dinov2": {
        "feature_backend": "hf",
        "feature_model": "facebook/dinov2-base",
    },
    "biomedclip": {
        "feature_backend": "openclip",
        "feature_model": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    },
    "derm_foundation": {
        "feature_backend": "keras",
        "feature_model": "google/derm-foundation",
        "keras_output_key": "embedding",
        "keras_image_size": 448,
    },
    "path_foundation": {
        "feature_backend": "keras",
        "feature_model": "google/path-foundation",
        "keras_output_key": "embedding",
        "keras_image_size": 224,
    },
    "chief": {
        "feature_backend": "chief",
        "feature_model": "CHIEF_CTransPath",
    },
    "uni": {
        "feature_backend": "timm",
        "feature_model": "hf-hub:MahmoodLab/UNI",
        "trust_remote_code": True,
    },
    "virchow": {
        "feature_backend": "timm",
        "feature_model": "hf-hub:paige-ai/Virchow",
        "trust_remote_code": True,
    },
    "medimageinsight": {
        "feature_backend": "medimageinsight",
        "feature_model": "lion-ai/MedImageInsights",
    },
}


def parse_dataset_value(value):
    value = str(value).lower()
    if value not in SUPPORTED_DATASETS:
        raise argparse.ArgumentTypeError(f"invalid dataset: {value}")
    return value


def parse_magnification_value(value):
    if value is None:
        return None
    value = str(value)
    if value.lower() in {"none", "null"}:
        return None
    if value not in {"40", "100", "200", "400"}:
        raise argparse.ArgumentTypeError(f"invalid BreakHis magnification: {value}")
    return value


def format_name_value(value):
    if isinstance(value, float):
        value = f"{value:g}"
    return str(value).replace("-", "m").replace(".", "p").replace("/", "-").replace("\\", "-").replace(":", "-")


def resolve_device(use_gpu, device_index):
    if not use_gpu:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("--gpu was requested, but CUDA is not available")
    return torch.device(f"cuda:{device_index}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dataset_samples(dataset):
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        base_dataset = dataset.dataset
        return [base_dataset.samples[index] for index in dataset.indices]
    return list(dataset.samples)


class PathDataset(Dataset):
    def __init__(self, samples):
        self.samples = [(str(path), int(label)) for path, label in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_paths(batch):
    paths, labels = zip(*batch)
    return list(paths), torch.tensor(labels, dtype=torch.long)


class HuggingFaceVisionExtractor:
    def __init__(self, model_id, device, trust_remote_code=False):
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "Hugging Face feature extraction requires transformers. "
                "Install transformers, or use --feature_backend torchvision."
            ) from exc

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        else:
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        features = pooled_features_from_outputs(outputs)
        return features.detach().float().cpu()


class TorchvisionFeatureExtractor:
    def __init__(self, model_name, device):
        from torchvision import models

        self.device = device
        weights = default_torchvision_weights(models, model_name)
        self.transform = weights.transforms()
        self.model, self.feature_dim = create_torchvision_feature_model(models, model_name, weights)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, images):
        batch = torch.stack([self.transform(image.convert("RGB")) for image in images]).to(self.device)
        features = self.model(batch)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        return features.detach().float().cpu()


class TorchvisionCheckpointFeatureExtractor(TorchvisionFeatureExtractor):
    def __init__(self, model_name, checkpoint_path, device):
        super().__init__(model_name, device)
        if not checkpoint_path:
            raise ValueError("--checkpoint_path is required for torchvision_checkpoint backend")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
        state_dict = normalize_state_dict_keys(state_dict)
        self.model.load_state_dict(state_dict, strict=False)


class OpenCLIPFeatureExtractor:
    def __init__(self, model_id, device):
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "OpenCLIP feature extraction requires open_clip_torch. "
                "Install open_clip_torch to use BiomedCLIP/CONCH-style backends."
            ) from exc

        self.device = device
        self.model, self.preprocess = open_clip.create_model_from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, images):
        batch = torch.stack([self.preprocess(image.convert("RGB")) for image in images]).to(self.device)
        features = self.model.encode_image(batch)
        if isinstance(features, (tuple, list)):
            features = features[0]
        return features.detach().float().cpu()


class TimmFeatureExtractor:
    def __init__(self, model_name, device, timm_kwargs=None):
        try:
            import timm
            from timm.data import create_transform, resolve_model_data_config
        except ImportError as exc:
            raise ImportError("timm feature extraction requires timm.") from exc

        self.device = device
        kwargs = {"pretrained": True, "num_classes": 0}
        kwargs.update(timm_kwargs or {})
        self.model = timm.create_model(model_name, **kwargs)
        config = resolve_model_data_config(self.model)
        self.transform = create_transform(**config, is_training=False)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, images):
        batch = torch.stack([self.transform(image.convert("RGB")) for image in images]).to(self.device)
        features = self.model(batch)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        return features.detach().float().cpu()


class KerasEmbeddingExtractor:
    def __init__(self, model_id, output_key="embedding", image_size=None):
        try:
            import tensorflow as tf
            from huggingface_hub import from_pretrained_keras
        except ImportError as exc:
            raise ImportError(
                "Keras medical foundation models require tensorflow and huggingface_hub."
            ) from exc

        self.tf = tf
        self.output_key = output_key
        self.image_size = image_size
        self.model = from_pretrained_keras(model_id)
        self.infer = self.model.signatures["serving_default"]

    def extract(self, images):
        serialized_inputs = [self.serialize_image(image) for image in images]
        outputs = self.infer(inputs=self.tf.constant(serialized_inputs))
        if self.output_key not in outputs:
            raise KeyError(f"Keras model output key {self.output_key!r} not found. Available: {sorted(outputs)}")
        return torch.from_numpy(outputs[self.output_key].numpy()).float()

    def serialize_image(self, image):
        image = image.convert("RGB")
        if self.image_size:
            image = image.resize((self.image_size, self.image_size))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return self.tf.train.Example(
            features=self.tf.train.Features(
                feature={
                    "image/encoded": self.tf.train.Feature(
                        bytes_list=self.tf.train.BytesList(value=[buffer.getvalue()])
                    )
                }
            )
        ).SerializeToString()


class ChiefFeatureExtractor:
    def __init__(self, model_path, device):
        if not model_path:
            raise ValueError("--checkpoint_path is required for chief backend")
        try:
            from CHIEF.models.ctran import ctranspath
            from torchvision import transforms
        except ImportError as exc:
            raise ImportError("CHIEF backend requires the local CHIEF package.") from exc

        self.device = device
        self.model = ctranspath()
        self.model.head = nn.Identity()
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @torch.no_grad()
    def extract(self, images):
        batch = torch.stack([self.transform(image.convert("RGB")) for image in images]).to(self.device)
        features = self.model(batch)
        if isinstance(features, (tuple, list)):
            features = features[0]
        return features.detach().float().cpu()


class MedImageInsightFeatureExtractor:
    def __init__(self, model_dir, vision_model_name, language_model_name):
        try:
            from medimageinsights import MedImageInsight
        except ImportError as first_exc:
            try:
                from MedImageInsights import MedImageInsight
            except ImportError:
                raise ImportError(
                    "MedImageInsight backend requires the MedImageInsights package or cloned HF repo."
                ) from first_exc

        self.model = MedImageInsight(
            model_dir=model_dir,
            vision_model_name=vision_model_name,
            language_model_name=language_model_name,
        )
        self.model.load_model()

    def extract(self, images):
        encoded_images = [base64.encodebytes(pil_image_bytes(image)).decode("utf-8") for image in images]
        outputs = self.model.encode(images=encoded_images)
        embeddings = outputs.get("image_embeddings", outputs.get("image_features"))
        if embeddings is None:
            raise KeyError(f"MedImageInsight image embeddings not found. Available: {sorted(outputs)}")
        return torch.as_tensor(embeddings, dtype=torch.float32)


def pooled_features_from_outputs(outputs):
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        return outputs.image_embeds
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        hidden = outputs.last_hidden_state
        if hidden.ndim == 3:
            return hidden[:, 0]
        return torch.flatten(hidden, 1)
    if isinstance(outputs, (tuple, list)) and outputs:
        first = outputs[0]
        if first.ndim == 3:
            return first[:, 0]
        return torch.flatten(first, 1)
    raise ValueError("Unable to infer pooled image features from model outputs")


def default_torchvision_weights(models_module, model_name):
    weight_name = {
        "resnet18": "ResNet18_Weights",
        "resnet34": "ResNet34_Weights",
        "resnet50": "ResNet50_Weights",
        "efficientnet_b0": "EfficientNet_B0_Weights",
        "efficientnet_b4": "EfficientNet_B4_Weights",
        "convnext_tiny": "ConvNeXt_Tiny_Weights",
        "vit_b_16": "ViT_B_16_Weights",
        "swin_t": "Swin_T_Weights",
    }.get(model_name)
    if weight_name is None:
        raise ValueError(f"Unsupported torchvision feature model: {model_name}")
    return getattr(models_module, weight_name).DEFAULT


def create_torchvision_feature_model(models_module, model_name, weights):
    model = getattr(models_module, model_name)(weights=weights)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim
    if hasattr(model, "heads") and hasattr(model.heads, "head") and isinstance(model.heads.head, nn.Linear):
        feature_dim = model.heads.head.in_features
        model.heads.head = nn.Identity()
        return model, feature_dim
    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):
            feature_dim = classifier.in_features
            model.classifier = nn.Identity()
            return model, feature_dim
        if isinstance(classifier, nn.Sequential):
            for index in range(len(classifier) - 1, -1, -1):
                layer = classifier[index]
                if isinstance(layer, nn.Linear):
                    feature_dim = layer.in_features
                    classifier[index] = nn.Identity()
                    return model, feature_dim
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        feature_dim = model.head.in_features
        model.head = nn.Identity()
        return model, feature_dim
    raise ValueError(f"Unable to remove classifier head for torchvision model: {model_name}")


def normalize_state_dict_keys(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        normalized[key] = value
    return normalized


def pil_image_bytes(image):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def parse_json_object(raw_value, argument_name):
    if not raw_value:
        return {}
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{argument_name} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{argument_name} must be a JSON object")
    return value


def apply_model_preset(args):
    if not args.model_preset:
        return args
    preset = MODEL_PRESETS[args.model_preset]
    for key, value in preset.items():
        setattr(args, key, value)
    return args


def create_feature_extractor(args, device):
    feature_backend = args.feature_backend
    feature_model = args.feature_model
    if feature_backend == "auto":
        if feature_model.startswith("torchvision:"):
            feature_backend = "torchvision"
        elif feature_model.startswith("openclip:"):
            feature_backend = "openclip"
        elif feature_model.startswith("timm:") or feature_model.startswith("hf-hub:"):
            feature_backend = "timm"
        else:
            feature_backend = "hf"
    if feature_backend == "torchvision":
        model_name = feature_model.split(":", 1)[1] if feature_model.startswith("torchvision:") else feature_model
        return TorchvisionFeatureExtractor(model_name, device)
    if feature_backend == "torchvision_checkpoint":
        model_name = feature_model.split(":", 1)[1] if feature_model.startswith("torchvision:") else feature_model
        return TorchvisionCheckpointFeatureExtractor(model_name, args.checkpoint_path, device)
    if feature_backend == "hf":
        model_id = feature_model.split(":", 1)[1] if feature_model.startswith("hf:") else feature_model
        return HuggingFaceVisionExtractor(model_id, device, trust_remote_code=args.trust_remote_code)
    if feature_backend == "openclip":
        model_id = feature_model.split(":", 1)[1] if feature_model.startswith("openclip:") else feature_model
        return OpenCLIPFeatureExtractor(model_id, device)
    if feature_backend == "timm":
        model_name = feature_model.split(":", 1)[1] if feature_model.startswith("timm:") else feature_model
        timm_kwargs = parse_json_object(args.timm_kwargs, "--timm_kwargs")
        return TimmFeatureExtractor(model_name, device, timm_kwargs=timm_kwargs)
    if feature_backend == "keras":
        return KerasEmbeddingExtractor(
            feature_model,
            output_key=args.keras_output_key,
            image_size=args.keras_image_size,
        )
    if feature_backend == "chief":
        return ChiefFeatureExtractor(args.checkpoint_path, device)
    if feature_backend == "medimageinsight":
        return MedImageInsightFeatureExtractor(
            model_dir=args.medimageinsight_model_dir,
            vision_model_name=args.medimageinsight_vision_model_name,
            language_model_name=args.medimageinsight_language_model_name,
        )
    raise ValueError(f"Unsupported feature backend: {feature_backend}")


def extract_features(samples, extractor, batch_size, num_workers=0):
    image_loader = load_default_image_loader()
    path_loader = DataLoader(
        PathDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_paths,
    )

    feature_batches = []
    label_batches = []
    for paths, labels in path_loader:
        images = [image_loader(path).convert("RGB") for path in paths]
        feature_batches.append(extractor.extract(images))
        label_batches.append(labels)
    return torch.cat(feature_batches, dim=0), torch.cat(label_batches, dim=0)


def standardize_train_test(train_features, test_features):
    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_features - mean) / std, (test_features - mean) / std


def compute_class_weights(labels, num_classes):
    counts = torch.bincount(labels, minlength=num_classes).float()
    counts = counts.clamp_min(1.0)
    return labels.numel() / (num_classes * counts)


class FeatureClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=0, dropout=0.0):
        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, features):
        return self.classifier(features)


def build_metrics(loss, labels, predictions):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def run_classifier_epoch(model, loader, criterion, optimizer, device):
    model.train(optimizer is not None)
    total_loss = 0.0
    labels = []
    predictions = []

    with torch.set_grad_enabled(optimizer is not None):
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * targets.size(0)
            labels.extend(targets.detach().cpu().tolist())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())

    return build_metrics(total_loss / len(loader.dataset), labels, predictions)


@dataclass
class FeatureTensors:
    train_features: torch.Tensor
    train_labels: torch.Tensor
    test_features: torch.Tensor
    test_labels: torch.Tensor


def train_feature_classifier(feature_tensors, num_classes, args, device):
    train_features = feature_tensors.train_features.float()
    test_features = feature_tensors.test_features.float()
    if args.standardize_features:
        train_features, test_features = standardize_train_test(train_features, test_features)

    train_dataset = TensorDataset(train_features, feature_tensors.train_labels.long())
    test_dataset = TensorDataset(test_features, feature_tensors.test_labels.long())
    train_loader = DataLoader(train_dataset, batch_size=args.classifier_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.classifier_batch_size, shuffle=False)

    classifier = FeatureClassifier(
        input_dim=train_features.shape[1],
        num_classes=num_classes,
        hidden_dim=args.classifier_hidden_dim,
        dropout=args.dropout,
    ).to(device)
    class_weights = None
    if args.class_weight == "balanced":
        class_weights = compute_class_weights(feature_tensors.train_labels.long(), num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_metrics = None
    best_f1 = -math.inf
    for epoch in range(1, args.num_epochs + 1):
        train_metrics = run_classifier_epoch(classifier, train_loader, criterion, optimizer, device)
        test_metrics = run_classifier_epoch(classifier, test_loader, criterion, None, device)
        if test_metrics["f1"] > best_f1:
            best_f1 = test_metrics["f1"]
            best_metrics = dict(test_metrics)
        if epoch == 1 or epoch == args.num_epochs or epoch % args.log_interval == 0:
            print(
                f"Epoch [{epoch}/{args.num_epochs}] - "
                f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}, "
                f"Test Loss: {test_metrics['loss']:.4f}, Test Acc: {test_metrics['accuracy']:.4f}, "
                f"Best F1: {best_f1:.4f}"
            )
    return classifier, best_metrics


def build_cache_path(args, trial_seed):
    model_name = format_name_value(args.feature_model)
    magnification = args.magnification if args.magnification is not None else "none"
    filename = (
        f"{args.dataset}_{format_name_value(args.test_split)}_{format_name_value(magnification)}_"
        f"{args.feature_backend}_{model_name}_seed{trial_seed}.pt"
    )
    return Path(args.feature_cache_dir) / filename


def load_or_extract_features(args, trial_seed, device):
    traintest_dataset, test_dataset, resize_size, _ = get_data(
        strategy="",
        dataset=args.dataset,
        magnification=args.magnification,
        dataroot=args.data_dir,
        random_state=trial_seed,
        test_split=args.test_split,
    )
    cache_path = build_cache_path(args, trial_seed)
    if args.feature_cache and cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        return FeatureTensors(
            train_features=cached["train_features"],
            train_labels=cached["train_labels"],
            test_features=cached["test_features"],
            test_labels=cached["test_labels"],
        )

    extractor = create_feature_extractor(args, device)
    train_samples = dataset_samples(traintest_dataset)
    test_samples = dataset_samples(test_dataset)
    print(f"Extracting features: train={len(train_samples)}, test={len(test_samples)}, resize={resize_size}")
    train_features, train_labels = extract_features(
        train_samples,
        extractor,
        batch_size=args.feature_batch_size,
        num_workers=args.num_workers,
    )
    test_features, test_labels = extract_features(
        test_samples,
        extractor,
        batch_size=args.feature_batch_size,
        num_workers=args.num_workers,
    )
    tensors = FeatureTensors(train_features, train_labels, test_features, test_labels)
    if args.feature_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "train_features": train_features,
                "train_labels": train_labels,
                "test_features": test_features,
                "test_labels": test_labels,
            },
            cache_path,
        )
        print(f"Saved feature cache: {cache_path}")
    return tensors


def build_stats_rows(records):
    stats = {}
    if not records:
        return stats
    metrics = sorted(records[0])
    for metric in metrics:
        values = np.asarray([record[metric] for record in records], dtype=float)
        stats[metric] = {
            "mean": f"{values.mean():.4f}",
            "std": f"{values.std():.4f}",
            "max": f"{values.max():.4f}",
            "min": f"{values.min():.4f}",
        }
    return stats


def write_stats_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = build_stats_rows(records)
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["metric", "mean", "std", "max", "min"])
        writer.writeheader()
        for metric, values in stats.items():
            writer.writerow({"metric": metric, **values})
    print(f"Saved summary: {path}")


def build_result_name(args):
    parts = [
        "foundation",
        args.dataset,
        format_name_value(args.test_split),
        format_name_value(args.feature_model),
    ]
    if args.dataset == "breakhis" and args.magnification is not None:
        parts.insert(2, f"{format_name_value(args.magnification)}X")
    if args.classifier_hidden_dim:
        parts.append(f"mlp{format_name_value(args.classifier_hidden_dim)}")
    else:
        parts.append("linear")
    parts.extend([f"ep{args.num_epochs}", f"lr{format_name_value(args.lr)}"])
    if args.class_weight != "none":
        parts.append(f"cw{args.class_weight}")
    return "_".join(parts)


def run_trial(args, trial_index):
    trial_seed = args.seed + trial_index
    set_seed(trial_seed)
    device = resolve_device(args.gpu, args.device)
    feature_tensors = load_or_extract_features(args, trial_seed, device)
    classifier, best_metrics = train_feature_classifier(feature_tensors, num_class(args.dataset), args, device)
    if args.save_classifier:
        save_dir = Path(args.classifier_save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{build_result_name(args)}_trial{trial_index + 1}.pth"
        torch.save({"model": classifier.state_dict(), "metrics": best_metrics}, save_path)
        print(f"Saved classifier: {save_path}")
    return best_metrics


def run_trials(args):
    records = []
    for trial_index in range(args.num_trials):
        print(f"\n{'=' * 50}\nTrial {trial_index + 1}/{args.num_trials}\n{'=' * 50}")
        records.append(run_trial(args, trial_index))
    write_stats_csv(Path(args.result_dir) / f"{build_result_name(args)}.csv", records)
    return records


def build_parser():
    parser = argparse.ArgumentParser(description="Frozen large-model feature extraction plus classifier training.")
    parser.add_argument("--data_dir", type=str, default="E:/Data/MedicalImage")
    parser.add_argument("--dataset", type=parse_dataset_value, default="pad-ufes-20")
    parser.add_argument("--magnification", type=parse_magnification_value, default=None)
    parser.add_argument("--test_split", type=float, default=0.3)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_preset", choices=sorted(MODEL_PRESETS), default="")
    parser.add_argument(
        "--feature_backend",
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
        default="auto",
    )
    parser.add_argument("--feature_model", default="facebook/dinov2-base")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--timm_kwargs", default="")
    parser.add_argument("--keras_output_key", default="embedding")
    parser.add_argument("--keras_image_size", type=int, default=None)
    parser.add_argument("--medimageinsight_model_dir", default="2024.09.27")
    parser.add_argument("--medimageinsight_vision_model_name", default="medimageinsigt-v1.0.0.pt")
    parser.add_argument("--medimageinsight_language_model_name", default="language_model.pth")
    parser.add_argument("--feature_batch_size", type=int, default=16)
    parser.add_argument("--classifier_batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--classifier_hidden_dim", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--class_weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--standardize_features", action="store_true", default=True)
    parser.add_argument("--no_standardize_features", dest="standardize_features", action="store_false")
    parser.add_argument("--feature_cache", action="store_true", default=True)
    parser.add_argument("--no_feature_cache", dest="feature_cache", action="store_false")
    parser.add_argument("--feature_cache_dir", default="feature_cache")
    parser.add_argument("--result_dir", default="result")
    parser.add_argument("--save_classifier", action="store_true")
    parser.add_argument("--classifier_save_dir", default="params_save/foundation_features")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    return parser


def main(argv=None):
    args = apply_model_preset(build_parser().parse_args(argv))
    os.makedirs(args.result_dir, exist_ok=True)
    return run_trials(args)


if __name__ == "__main__":
    main()
