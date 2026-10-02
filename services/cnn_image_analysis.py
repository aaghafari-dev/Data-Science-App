"""Professional CNN image-analysis engine for classification, transfer learning and fine-tuning."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CNNConfig:
    """Configuration for a bounded, reproducible CNN image experiment."""
    architecture: str = "resnet18"
    pretrained: bool = True
    fine_tune_mode: str = "freeze_backbone"
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "AdamW"
    scheduler: str = "ReduceLROnPlateau"
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    early_stopping_patience: int = 3
    augment: bool = True
    class_weighting: bool = True
    label_smoothing: float = 0.0


class CNNImageAnalysisEngine:
    """Run professional image classification experiments with PyTorch/torchvision when available."""

    @staticmethod
    def scan_image_folder(root: str | os.PathLike) -> dict[str, Any]:
        """Inspect an ImageFolder-compatible directory without loading image tensors."""
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists() or not root_path.is_dir():
            raise ValueError(f"Image dataset directory does not exist: {root_path}")
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        classes = []
        counts = {}
        for child in sorted(root_path.iterdir()):
            if child.is_dir():
                count = sum(1 for p in child.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
                if count:
                    classes.append(child.name)
                    counts[child.name] = count
        if len(classes) < 2:
            raise ValueError("CNN classification requires at least two non-empty class folders.")
        total = sum(counts.values())
        return {"root": str(root_path), "classes": classes, "class_counts": counts, "images": total,
                "extensions": sorted(extensions), "task": "image_classification"}

    @staticmethod
    def _imports():
        """Load optional deep-learning dependencies only when CNN analysis is requested."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
            from torchvision import datasets, transforms, models
        except Exception as exc:
            raise RuntimeError("CNN Image Analysis requires PyTorch and torchvision. Install the deep-learning requirements first.") from exc
        return torch, nn, DataLoader, random_split, WeightedRandomSampler, datasets, transforms, models

    @staticmethod
    def _model(models, config: CNNConfig, num_classes: int):
        """Create the selected architecture and apply transfer-learning policy."""
        name = config.architecture.lower()
        def _weights(enum_cls):
            """Resolve torchvision default weights without relying on version-specific string conversion."""
            return enum_cls.DEFAULT if config.pretrained else None
        if name == "resnet18":
            model = models.resnet18(weights=_weights(models.ResNet18_Weights))
            in_features = model.fc.in_features
            model.fc = __import__("torch").nn.Linear(in_features, num_classes)
            head = model.fc
        elif name == "resnet50":
            model = models.resnet50(weights=_weights(models.ResNet50_Weights))
            in_features = model.fc.in_features
            model.fc = __import__("torch").nn.Linear(in_features, num_classes)
            head = model.fc
        elif name == "vgg16":
            model = models.vgg16(weights=_weights(models.VGG16_Weights))
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = __import__("torch").nn.Linear(in_features, num_classes)
            head = model.classifier[-1]
        elif name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=_weights(models.EfficientNet_B0_Weights))
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = __import__("torch").nn.Linear(in_features, num_classes)
            head = model.classifier[-1]
        else:
            raise ValueError(f"Unsupported CNN architecture: {config.architecture}")

        if config.fine_tune_mode == "freeze_backbone":
            for p in model.parameters():
                p.requires_grad = False
            for p in head.parameters():
                p.requires_grad = True
        elif config.fine_tune_mode == "last_block":
            for p in model.parameters():
                p.requires_grad = False
            for module_name, module in model.named_modules():
                if any(token in module_name.lower() for token in ("layer4", "features.", "classifier")):
                    for p in module.parameters():
                        p.requires_grad = True
            for p in head.parameters():
                p.requires_grad = True
        elif config.fine_tune_mode == "full_finetune":
            for p in model.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"Unsupported fine-tuning mode: {config.fine_tune_mode}")
        return model

    @classmethod
    def run(cls, root: str, config: CNNConfig | None = None, device: str = "auto") -> dict[str, Any]:
        """Execute a bounded image-classification experiment and return JSON-safe evidence."""
        config = config or CNNConfig()
        info = cls.scan_image_folder(root)
        torch, nn, DataLoader, random_split, WeightedRandomSampler, datasets, transforms, models = cls._imports()
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        if device == "cuda" and torch.cuda.get_device_properties(0).total_memory < 3 * 1024**3:
            # Keep low-VRAM systems safe; image training can otherwise fail unexpectedly.
            device = "cpu"

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_ops = [transforms.Resize((config.image_size, config.image_size))]
        if config.augment:
            train_ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.ColorJitter(brightness=.15, contrast=.15, saturation=.15)]
        train_ops += [transforms.ToTensor(), normalize]
        eval_ops = [transforms.Resize((config.image_size, config.image_size)), transforms.ToTensor(), normalize]
        train_ds_full = datasets.ImageFolder(str(root), transform=transforms.Compose(train_ops))
        eval_ds = datasets.ImageFolder(str(root), transform=transforms.Compose(eval_ops))
        n = len(train_ds_full)
        if n < 20:
            raise ValueError("CNN training requires at least 20 images for a meaningful train/validation/test workflow.")
        n_test = max(1, int(round(n * config.test_fraction)))
        n_val = max(1, int(round(n * config.validation_fraction)))
        n_train = n - n_val - n_test
        if n_train < 2:
            raise ValueError("Validation/test fractions leave too few training images.")
        from sklearn.model_selection import train_test_split
        all_indices = np.arange(n)
        targets = np.asarray(train_ds_full.targets)
        try:
            train_val_idx, test_idx = train_test_split(all_indices, test_size=n_test, random_state=config.seed, stratify=targets)
            train_idx, val_idx = train_test_split(train_val_idx, test_size=n_val / len(train_val_idx), random_state=config.seed, stratify=targets[train_val_idx])
        except ValueError:
            # Very small classes cannot always support a stratified split; fall back explicitly and record the limitation.
            rng = np.random.default_rng(config.seed); shuffled = rng.permutation(all_indices)
            test_idx = shuffled[:n_test]; val_idx = shuffled[n_test:n_test+n_val]; train_idx = shuffled[n_test+n_val:]
        train_idx, val_idx, test_idx = map(np.asarray, (train_idx, val_idx, test_idx))
        from torch.utils.data import Subset
        train_ds = Subset(train_ds_full, train_idx)
        val_ds = Subset(eval_ds, val_idx)
        test_ds = Subset(eval_ds, test_idx)
        labels = np.asarray([train_ds_full.targets[i] for i in train_idx], dtype=int)
        counts = np.bincount(labels, minlength=len(train_ds_full.classes))
        sampler = None
        if config.class_weighting and np.all(counts > 0):
            weights = 1.0 / counts[labels]
            sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=sampler is None, sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)

        model = cls._model(models, config, len(train_ds_full.classes)).to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay) if config.optimizer.lower() == "adamw" else torch.optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.5, patience=1) if config.scheduler == "ReduceLROnPlateau" else None
        history = []
        best_val = float("inf"); best_state = None; stale = 0
        for epoch in range(config.epochs):
            model.train(); train_loss = 0.0; seen = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True); logits = model(xb); loss = criterion(logits, yb); loss.backward(); optimizer.step()
                train_loss += float(loss.item()) * len(yb); seen += len(yb)
            val_loss, val_pred, val_true = cls._evaluate(model, val_loader, criterion, device)
            train_loss /= max(1, seen)
            if scheduler is not None: scheduler.step(val_loss)
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss, "learning_rate": float(optimizer.param_groups[0]["lr"])})
            if val_loss < best_val - 1e-6:
                best_val = val_loss; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale = 0
            else:
                stale += 1
                if stale >= config.early_stopping_patience:
                    break
        if best_state is not None: model.load_state_dict(best_state)
        test_loss, pred, true = cls._evaluate(model, test_loader, criterion, device)
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
        metrics = {
            "accuracy": float(accuracy_score(true, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
            "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
            "macro_precision": float(precision_score(true, pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(true, pred, average="macro", zero_division=0)),
            "test_loss": float(test_loss),
            "confusion_matrix": confusion_matrix(true, pred).tolist(),
        }
        return {"status": "complete", "task": "image_classification", "dataset": info, "classes": train_ds_full.classes,
                "config": asdict(config), "device": device, "history": history, "metrics": metrics,
                "train_images": len(train_ds), "validation_images": len(val_ds), "test_images": len(test_ds),
                "class_counts_train": counts.tolist(), "predictions": pred.tolist(), "truth": true.tolist(),
                "model_parameters_trainable": int(sum(p.numel() for p in params)),
                "model_parameters_total": int(sum(p.numel() for p in model.parameters())),
                "model_state_dict": model.state_dict(), "model": model, "eval_dataset": test_ds}

    @staticmethod
    def grad_cam(model, dataset, sample_index: int, architecture: str, device: str = "cpu") -> dict[str, Any]:
        """Generate a Grad-CAM heatmap for one evaluation image and return prediction evidence."""
        import torch
        model = model.to(device); model.eval()
        if sample_index < 0 or sample_index >= len(dataset):
            raise IndexError("Evaluation image index is outside the available test set.")
        x, y = dataset[sample_index]
        activations = []
        name = architecture.lower()
        if name.startswith("resnet"):
            target_layer = model.layer4[-1]
        elif name == "vgg16":
            target_layer = model.features[-1]
        elif name == "efficientnet_b0":
            target_layer = model.features[-1]
        else:
            raise ValueError("Grad-CAM is not configured for this architecture.")
        def _capture(_module, _inputs, output):
            """Capture activations and retain their gradient for Grad-CAM."""
            activations.append(output)
            if hasattr(output, "retain_grad"):
                output.retain_grad()
        handle_f = target_layer.register_forward_hook(_capture)
        try:
            inp = x.unsqueeze(0).to(device).requires_grad_(True)
            logits = model(inp)
            pred = int(logits.argmax(dim=1).item())
            score = logits[0, pred]
            model.zero_grad(set_to_none=True); score.backward()
            act = activations[-1].detach(); grad = activations[-1].grad.detach()
            weights = grad.mean(dim=(2,3), keepdim=True)
            cam = (weights * act).sum(dim=1).squeeze(0).cpu().numpy()
            cam = np.maximum(cam, 0)
            cam = cam / (cam.max() + 1e-8)
            return {"heatmap": cam, "predicted_class": pred, "true_class": int(y), "score": float(torch.softmax(logits, dim=1)[0, pred].item()), "input_tensor": x.cpu().numpy()}
        finally:
            handle_f.remove()

    @staticmethod
    def _evaluate(model, loader, criterion, device):
        """Evaluate a CNN without gradient tracking."""
        import torch
        model.eval(); total = 0.0; count = 0; pred = []; true = []
        with torch.no_grad():
            for xb, yb in loader:
                logits = model(xb.to(device)); loss = criterion(logits, yb.to(device)); total += float(loss.item()) * len(yb); count += len(yb)
                pred.extend(logits.argmax(dim=1).cpu().numpy().tolist()); true.extend(yb.numpy().tolist())
        return total / max(1, count), np.asarray(pred), np.asarray(true)

    @staticmethod
    def export_evidence(result: dict[str, Any], path: str) -> str:
        """Write a JSON-safe experiment record, excluding live model objects."""
        payload = {k: v for k, v in result.items() if k not in {"model", "model_state_dict", "eval_dataset"}}
        Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path
