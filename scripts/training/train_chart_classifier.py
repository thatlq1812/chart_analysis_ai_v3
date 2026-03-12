"""
Chart Type Classifier Training - Unified Script

Trains a torchvision backbone for chart type classification.
Replaces train_resnet18_v2.py and train_resnet_4class.py.

Modes (set via config or --override):
    4class  bar | line | pie | others      (thesis scope, default)
    8class  area | bar | box | heatmap | histogram | line | pie | scatter

Supported backbones (classifier.model.backbone):
    resnet18  resnet34  resnet50
    efficientnet_b0  efficientnet_b1
    mobilenet_v3_small  mobilenet_v3_large

Config: config/training.yaml  →  classifier.*

Usage:
    # Default (4-class ResNet-18):
    python scripts/training/train_chart_classifier.py

    # EfficientNet-B0 ablation:
    python scripts/training/train_chart_classifier.py \\
        --override classifier.model.backbone=efficientnet_b0 \\
        --override classifier.output.model_name=efficientnet_b0_4class_v3

    # 8-class research run:
    python scripts/training/train_chart_classifier.py \\
        --override classifier.mode=8class \\
        --override classifier.output.model_name=resnet18_8class_v3

    # Quick smoke-test (no GPU needed):
    python scripts/training/train_chart_classifier.py \\
        --override classifier.data.max_samples_per_class=200 \\
        --override classifier.training.epochs=3 \\
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm
import PIL.Image
from PIL import Image

PIL.Image.MAX_IMAGE_PIXELS = 200_000_000

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import DictConfig, OmegaConf

from src.training.run_manager import RunManager
from src.training.experiment_tracker import ExperimentTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Backbone Registry
# ===========================================================================

@dataclass
class BackboneSpec:
    """Specifies how to build and modify a torchvision backbone."""
    factory: Callable
    weights: Any
    head_attr: str          # Attribute name on the model for the classifier head
    in_features_fn: Callable  # model -> int

    def build(self, pretrained: bool):
        return self.factory(weights=self.weights if pretrained else None)

    def in_features(self, model: nn.Module) -> int:
        return self.in_features_fn(model)

    def freeze_all_except_head(self, model: nn.Module) -> None:
        head = getattr(model, self.head_attr)
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith(self.head_attr)
        logger.info(
            f"Backbone frozen | trainable_head={self.head_attr}"
        )

    def unfreeze_all(self, model: nn.Module) -> None:
        for param in model.parameters():
            param.requires_grad = True
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Backbone unfrozen | trainable_params={total:,}")


def _resnet_in_features(m: nn.Module) -> int:
    return m.fc.in_features

def _eff_or_mobile_in_features(m: nn.Module) -> int:
    # EfficientNet / MobileNetV3: classifier is nn.Sequential, last Linear
    for layer in reversed(list(m.classifier.children())):
        if isinstance(layer, nn.Linear):
            return layer.in_features
    raise RuntimeError("Cannot determine in_features for classifier head")


BACKBONE_REGISTRY: Dict[str, BackboneSpec] = {
    "resnet18": BackboneSpec(
        factory=models.resnet18,
        weights=models.ResNet18_Weights.IMAGENET1K_V1,
        head_attr="fc",
        in_features_fn=_resnet_in_features,
    ),
    "resnet34": BackboneSpec(
        factory=models.resnet34,
        weights=models.ResNet34_Weights.IMAGENET1K_V1,
        head_attr="fc",
        in_features_fn=_resnet_in_features,
    ),
    "resnet50": BackboneSpec(
        factory=models.resnet50,
        weights=models.ResNet50_Weights.IMAGENET1K_V2,
        head_attr="fc",
        in_features_fn=_resnet_in_features,
    ),
    "efficientnet_b0": BackboneSpec(
        factory=models.efficientnet_b0,
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        head_attr="classifier",
        in_features_fn=_eff_or_mobile_in_features,
    ),
    "efficientnet_b1": BackboneSpec(
        factory=models.efficientnet_b1,
        weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1,
        head_attr="classifier",
        in_features_fn=_eff_or_mobile_in_features,
    ),
    "mobilenet_v3_small": BackboneSpec(
        factory=models.mobilenet_v3_small,
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        head_attr="classifier",
        in_features_fn=_eff_or_mobile_in_features,
    ),
    "mobilenet_v3_large": BackboneSpec(
        factory=models.mobilenet_v3_large,
        weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        head_attr="classifier",
        in_features_fn=_eff_or_mobile_in_features,
    ),
}


# ===========================================================================
# Dataset
# ===========================================================================

class ClassifierDataset(Dataset):
    """
    Chart classification dataset supporting both 4-class and 8-class modes.

    4class mode: applies a remap dict to merge raw folders into target classes.
    8class mode: uses raw folder names directly; unlisted folders are skipped.

    Args:
        classified_dir:        Path to classified_charts/ root.
        split:                 'train' | 'val' | 'test'.
        transform:             torchvision transforms.
        classes:               Ordered list of class names (defines label indices).
        remap:                 Dict mapping raw folder name → target class.
                               Pass {} for 8class mode (identity mapping).
        skip_unlisted:         If True, folders absent from remap are skipped.
                               If False, they are assigned to 'others'.
        split_ratio:           (train, val, test) fractions; must sum to 1.
        seed:                  Random seed.
        max_samples_per_class: Optional cap per TARGET class (train split only).
        image_extensions:      File suffixes to scan.
    """

    def __init__(
        self,
        classified_dir: Path,
        split: str,
        transform,
        classes: List[str],
        remap: Dict[str, str],
        skip_unlisted: bool = False,
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        max_samples_per_class: Optional[int] = None,
        image_extensions: Optional[List[str]] = None,
    ) -> None:
        self.classified_dir = Path(classified_dir)
        self.split = split
        self.transform = transform
        self.classes = classes
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(classes)}
        self.remap = remap
        self.skip_unlisted = skip_unlisted
        self.max_cap = max_samples_per_class
        self._exts = set(image_extensions or [".png", ".jpg", ".jpeg", ".webp"])

        random.seed(seed)
        np.random.seed(seed)

        self.samples: List[Tuple[Path, str]] = self._load(split_ratio)

        logger.info(
            f"ClassifierDataset | split={split} | samples={len(self.samples)} | "
            f"dist={self.class_distribution()}"
        )

    def _load(self, split_ratio: Tuple) -> List[Tuple[Path, str]]:
        all_samples: List[Tuple[Path, str]] = []

        for folder in sorted(self.classified_dir.iterdir()):
            if not folder.is_dir():
                continue

            raw_name = folder.name
            if raw_name in self.remap:
                target = self.remap[raw_name]
            elif self.skip_unlisted:
                continue
            elif "others" in self.classes:
                target = "others"
            else:
                continue

            # Skip if target class not in our class list
            if target not in self.class_to_idx:
                continue

            images = [
                p for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in self._exts
            ]
            if not images:
                continue

            random.shuffle(images)
            n = len(images)
            t_end = int(n * split_ratio[0])
            v_end = t_end + int(n * split_ratio[1])

            if self.split == "train":
                selected = images[:t_end]
            elif self.split == "val":
                selected = images[t_end:v_end]
            else:
                selected = images[v_end:]

            for p in selected:
                all_samples.append((p, target))

        # Per-class cap (train only)
        if self.max_cap and self.split == "train":
            per: Dict[str, List] = {c: [] for c in self.classes}
            for path, cls in all_samples:
                per[cls].append((path, cls))
            capped: List[Tuple[Path, str]] = []
            for items in per.values():
                random.shuffle(items)
                capped.extend(items[: self.max_cap])
            all_samples = capped

        random.shuffle(all_samples)
        return all_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, cls = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[cls]

    def class_distribution(self) -> Dict[str, int]:
        return dict(sorted(Counter(c for _, c in self.samples).items()))

    def class_weights(self) -> torch.Tensor:
        dist = self.class_distribution()
        total = sum(dist.values())
        return torch.tensor(
            [total / (len(self.classes) * dist.get(c, 1)) for c in self.classes],
            dtype=torch.float32,
        )

    def sample_weights(self) -> List[float]:
        dist = self.class_distribution()
        return [1.0 / dist[cls] for _, cls in self.samples]


def build_datasets(cfg: DictConfig) -> Tuple[
    ClassifierDataset, ClassifierDataset, ClassifierDataset
]:
    """Build train/val/test datasets from config."""
    mode: str = cfg.classifier.mode
    data_cfg = cfg.classifier.data
    classified_dir = PROJECT_ROOT / data_cfg.classified_dir
    split_ratio: Tuple = tuple(data_cfg.split_ratio)
    exts: List[str] = list(data_cfg.image_extensions)
    seed: int = int(data_cfg.seed)
    max_cap = OmegaConf.select(data_cfg, "max_samples_per_class", default=None)
    image_size: int = int(cfg.classifier.model.image_size)

    if mode == "3class":
        classes: List[str] = list(cfg.classifier.classes_3class)
        remap: Dict[str, str] = OmegaConf.to_container(
            cfg.classifier.remap_3class, resolve=True
        )
        skip_unlisted = True   # folders absent from remap_3class are ignored
    elif mode == "4class":
        classes = list(cfg.classifier.classes_4class)
        remap = OmegaConf.to_container(cfg.classifier.remap_4class, resolve=True)
        skip_unlisted = False  # unlisted → others
    elif mode == "8class":
        classes = list(cfg.classifier.classes_8class)
        remap = {c: c for c in classes}  # identity mapping
        skip_unlisted = True              # unlisted folders ignored
    else:
        raise ValueError(
            f"Unknown classifier.mode='{mode}'. Expected '3class', '4class' or '8class'."
        )

    train_tf = _transforms(image_size, training=True)
    eval_tf = _transforms(image_size, training=False)

    common = dict(
        classified_dir=classified_dir,
        classes=classes,
        remap=remap,
        skip_unlisted=skip_unlisted,
        split_ratio=split_ratio,
        seed=seed,
        image_extensions=exts,
    )

    train_ds = ClassifierDataset(
        split="train", transform=train_tf, max_samples_per_class=max_cap, **common
    )
    val_ds = ClassifierDataset(split="val", transform=eval_tf, **common)
    test_ds = ClassifierDataset(split="test", transform=eval_tf, **common)

    return train_ds, val_ds, test_ds


# ===========================================================================
# Transforms
# ===========================================================================

def _transforms(image_size: int = 224, training: bool = True):
    resize_to = image_size + 32  # always resize 32px larger, then crop
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if training:
        return transforms.Compose([
            transforms.Resize(resize_to),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),   # Help pie generalize; 15deg safe for bar/line
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize(resize_to),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])


# ===========================================================================
# Model
# ===========================================================================

class ClassifierModel(nn.Module):
    """
    Unified chart type classifier built on any registered torchvision backbone.

    Replaces the backbone's classification head with:
        Dropout(dropout) → Linear(in_features, num_classes)

    Supports freeze/unfreeze for two-phase training.
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if backbone_name not in BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Supported: {sorted(BACKBONE_REGISTRY.keys())}"
            )
        spec = BACKBONE_REGISTRY[backbone_name]
        self._spec = spec
        self.net = spec.build(pretrained)

        in_f = spec.in_features(self.net)
        new_head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_f, num_classes))
        setattr(self.net, spec.head_attr, new_head)

        params = sum(p.numel() for p in self.parameters()) / 1e6
        logger.info(
            f"ClassifierModel | backbone={backbone_name} | "
            f"num_classes={num_classes} | params={params:.2f}M | "
            f"pretrained={pretrained}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def freeze_backbone(self) -> None:
        self._spec.freeze_all_except_head(self.net)

    def unfreeze_backbone(self) -> None:
        self._spec.unfreeze_all(self.net)


# ===========================================================================
# Training helpers
# ===========================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    log_interval: int = 50,
) -> Tuple[float, float]:
    model.train()
    running_loss = total = correct = 0

    pbar = tqdm(loader, desc="  train", leave=False, dynamic_ncols=True)
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        preds = logits.argmax(dim=1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

        if i % log_interval == 0:
            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                acc=f"{100.0 * correct / max(total, 1):.1f}%",
            )

    return running_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    classes: List[str],
) -> Tuple[float, float, Dict[str, Any]]:
    model.eval()
    n_cls = len(classes)
    running_loss = total = correct = 0
    class_correct: Counter = Counter()
    class_total: Counter = Counter()
    all_preds: List[int] = []
    all_labels: List[int] = []

    for images, labels in tqdm(loader, desc="  eval ", leave=False, dynamic_ncols=True):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item()
        preds = logits.argmax(dim=1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()
        for lbl, pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
            class_total[lbl] += 1
            if lbl == pred:
                class_correct[lbl] += 1
            all_labels.append(lbl)
            all_preds.append(pred)

    # Confusion matrix
    cm = [[0] * n_cls for _ in range(n_cls)]
    for t, p in zip(all_labels, all_preds):
        cm[t][p] += 1

    # Per-class precision / recall / F1
    per_acc: Dict[str, float] = {}
    per_f1: Dict[str, float] = {}
    for i, cls in enumerate(classes):
        n = class_total.get(i, 0)
        per_acc[cls] = class_correct.get(i, 0) / n * 100.0 if n else 0.0
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n_cls)) - tp
        fn = sum(cm[i][c] for c in range(n_cls)) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_f1[cls] = round(f1 * 100.0, 2)

    macro_f1 = sum(per_f1.values()) / n_cls

    stats: Dict[str, Any] = {
        "per_class_accuracy": per_acc,
        "per_class_f1": per_f1,
        "macro_f1": round(macro_f1, 2),
        "confusion_matrix": cm,
        "class_names": classes,
    }
    return running_loss / len(loader), 100.0 * correct / total, stats


# ===========================================================================
# Optimizer / scheduler factories
# ===========================================================================

def _make_optimizer_scheduler(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    epochs_remaining: int,
    warmup_epochs: int,
) -> Tuple[optim.Optimizer, Any, Optional[Any]]:
    """
    Build Adam optimizer + CosineAnnealingLR + optional linear warmup.

    Returns:
        (optimizer, cosine_scheduler, warmup_scheduler | None)
    """
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs_remaining, eta_min=1e-6
    )
    warmup = (
        optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        if warmup_epochs > 0
        else None
    )
    return optimizer, cosine, warmup


# ===========================================================================
# Main training loop
# ===========================================================================

def train(
    cfg: DictConfig,
    run_manager: RunManager,
    tracker: ExperimentTracker,
    dry_run: bool = False,
) -> Dict[str, Any]:
    tcfg = cfg.classifier.training
    mcfg = cfg.classifier.model
    ocfg = cfg.classifier.output
    ecfg = cfg.classifier.evaluation

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(
        f"Device: {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    )

    # --- Build datasets ---
    train_ds, val_ds, test_ds = build_datasets(cfg)
    classes = train_ds.classes

    if dry_run:
        logger.info("Dry-run mode: datasets built, skipping training.")
        return {"dry_run": True, "classes": classes, "train_samples": len(train_ds)}

    # --- DataLoaders ---
    nw = int(tcfg.num_workers)
    bs = int(tcfg.batch_size)
    pw = bool(OmegaConf.select(tcfg, "persistent_workers", default=False)) and nw > 0
    pf = int(OmegaConf.select(tcfg, "prefetch_factor", default=2)) if nw > 0 else None
    sw = train_ds.sample_weights()
    sampler = WeightedRandomSampler(weights=sw, num_samples=len(sw), replacement=True)

    logger.info(
        f"DataLoader | batch_size={bs} | num_workers={nw} | "
        f"persistent_workers={pw} | prefetch_factor={pf}"
    )

    train_loader = DataLoader(
        train_ds, batch_size=bs, sampler=sampler,
        num_workers=nw, pin_memory=True, drop_last=True,
        persistent_workers=pw, prefetch_factor=pf,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True,
        persistent_workers=pw, prefetch_factor=pf,
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        num_workers=nw, pin_memory=True,
        persistent_workers=pw, prefetch_factor=pf,
    )

    # --- Model ---
    model = ClassifierModel(
        backbone_name=mcfg.backbone,
        num_classes=len(classes),
        pretrained=bool(mcfg.pretrained),
        dropout=float(mcfg.dropout),
    ).to(device)

    # --- Loss ---
    cw = train_ds.class_weights().to(device)
    label_smoothing = float(OmegaConf.select(tcfg, "label_smoothing", default=0.0))
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
    logger.info(f"Loss | label_smoothing={label_smoothing} | class_weights={cw.tolist()}")

    # --- AMP ---
    use_amp = bool(tcfg.use_amp) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    epochs: int = int(tcfg.epochs)
    freeze_epochs: int = int(tcfg.freeze_backbone_epochs)
    warmup_epochs: int = int(tcfg.warmup_epochs)
    wd: float = float(tcfg.weight_decay)
    log_interval: int = int(tcfg.log_interval_batches)
    ckpt_dir = run_manager.checkpoints_dir

    best_val_acc = 0.0
    best_ckpt: Optional[Path] = None
    history: List[Dict[str, Any]] = []
    optimizer = cosine_sched = warmup_sched = None

    patience: int = int(OmegaConf.select(tcfg, "early_stopping_patience", default=0))
    no_improve: int = 0  # epochs without val_acc improvement in current phase
    early_stopped: bool = False

    for epoch in range(1, epochs + 1):
        # Phase 1 → Phase 2 transition
        if epoch == 1:
            model.freeze_backbone()
            optimizer, cosine_sched, warmup_sched = _make_optimizer_scheduler(
                model, float(tcfg.frozen_lr), wd,
                epochs_remaining=freeze_epochs, warmup_epochs=warmup_epochs,
            )
            logger.info(
                f"Phase 1: frozen backbone | epochs=1-{freeze_epochs} | "
                f"lr={tcfg.frozen_lr}"
            )
        elif epoch == freeze_epochs + 1:
            model.unfreeze_backbone()
            optimizer, cosine_sched, warmup_sched = _make_optimizer_scheduler(
                model, float(tcfg.learning_rate), wd,
                epochs_remaining=epochs - freeze_epochs,
                warmup_epochs=warmup_epochs,
            )
            logger.info(
                f"Phase 2: full fine-tune | epochs={freeze_epochs + 1}-{epochs} | "
                f"lr={tcfg.learning_rate}"
            )
            no_improve = 0  # Reset counter when phase changes

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch}/{epochs} | lr={current_lr:.2e}")

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp, log_interval,
        )
        val_loss, val_acc, val_stats = evaluate(model, val_loader, criterion, device, classes)

        # Scheduler step
        in_warmup = (
            (epoch <= warmup_epochs and epoch <= freeze_epochs)
            or (
                epoch > freeze_epochs
                and (epoch - freeze_epochs) <= warmup_epochs
            )
        )
        if warmup_sched and in_warmup:
            warmup_sched.step()
        else:
            cosine_sched.step()

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "val_macro_f1": val_stats["macro_f1"],
            "lr": round(current_lr, 8),
        }
        history.append(row)

        tracker.log_metrics(
            {
                "train/loss": train_loss, "train/acc": train_acc,
                "val/loss": val_loss, "val/acc": val_acc,
                "val/macro_f1": val_stats["macro_f1"], "lr": current_lr,
            },
            step=epoch,
        )

        logger.info(
            f"  train loss={train_loss:.4f} acc={train_acc:.1f}% | "
            f"val loss={val_loss:.4f} acc={val_acc:.1f}% "
            f"macro_f1={val_stats['macro_f1']:.1f}%"
        )

        # Checkpoint + early stopping counter
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            best_ckpt = ckpt_dir / f"epoch_{epoch:03d}_valacc_{val_acc:.1f}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "val_macro_f1": val_stats["macro_f1"],
                    "classes": classes,
                    "backbone": mcfg.backbone,
                    "mode": cfg.classifier.mode,
                    "config": OmegaConf.to_container(cfg.classifier, resolve=True),
                },
                best_ckpt,
            )
            logger.info(
                f"  New best | val_acc={val_acc:.2f}% | ckpt={best_ckpt.name}"
            )
        else:
            no_improve += 1
            if patience > 0:
                logger.info(
                    f"  No improvement | patience={no_improve}/{patience}"
                )
            if patience > 0 and no_improve >= patience and epoch > freeze_epochs:
                logger.info(
                    f"Early stopping triggered | epoch={epoch} | "
                    f"best_val_acc={best_val_acc:.2f}% | "
                    f"no_improve={no_improve}"
                )
                early_stopped = True
                break

    # --- Test evaluation ---
    logger.info("Running final test evaluation...")
    if best_ckpt and best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded best weights | epoch={ckpt['epoch']} val_acc={ckpt['val_acc']:.2f}%")

    test_loss, test_acc, test_stats = evaluate(
        model, test_loader, criterion, device, classes
    )

    # --- Save final weights ---
    weights_dir = PROJECT_ROOT / ocfg.weights_dir
    weights_dir.mkdir(parents=True, exist_ok=True)
    final_weights = weights_dir / f"{ocfg.model_name}_best.pt"
    torch.save(model.state_dict(), final_weights)
    logger.info(f"Final weights saved: {final_weights}")

    # --- Thesis pass check ---
    mode = cfg.classifier.mode
    target_acc = float(ecfg.target_accuracy) * 100
    target_f1 = float(ecfg.target_f1_macro) * 100
    target_f1_others = float(ecfg.target_f1_others) * 100
    others_f1 = test_stats["per_class_f1"].get("others", 100.0)  # 100 = n/a for 8class

    thesis_pass = (
        test_acc >= target_acc
        and test_stats["macro_f1"] >= target_f1
        and (mode not in ("3class", "4class") or mode == "3class" or others_f1 >= target_f1_others)
    )

    results: Dict[str, Any] = {
        "run_name": run_manager.run_name,
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "backbone": mcfg.backbone,
        "classes": classes,
        "num_classes": len(classes),
        "device": str(device),
        "epochs_trained": epoch,
        "early_stopped": early_stopped,
        "best_val_acc": round(best_val_acc, 2),
        "test_accuracy": round(test_acc, 2),
        "test_macro_f1": test_stats["macro_f1"],
        "test_per_class_accuracy": test_stats["per_class_accuracy"],
        "test_per_class_f1": test_stats["per_class_f1"],
        "test_confusion_matrix": test_stats["confusion_matrix"],
        "thesis_targets": {
            "target_accuracy": target_acc,
            "target_f1_macro": target_f1,
            "target_f1_others": target_f1_others,
        },
        "thesis_pass": thesis_pass,
        "weights_path": str(final_weights),
        "training_history": history,
    }

    results_path = weights_dir / f"{ocfg.model_name}_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved: {results_path}")

    _print_benchmark(results)

    tracker.log_metrics({"test/acc": test_acc, "test/macro_f1": test_stats["macro_f1"]}, step=epochs)
    tracker.finish()

    return results


def _print_benchmark(r: Dict[str, Any]) -> None:
    w = 58
    sep = "-" * w
    print(f"\n{'=' * w}")
    print(f"  BENCHMARK  |  {r['backbone']}  |  mode={r['mode']}  |  {r['num_classes']} classes")
    print(f"{'=' * w}")
    print(f"  Run: {r['run_name']}")
    targets = r["thesis_targets"]
    checks = [
        ("Test Accuracy (%)",  r["test_accuracy"],  targets["target_accuracy"]),
        ("Macro F1 (%)",       r["test_macro_f1"],  targets["target_f1_macro"]),
    ]
    if r["mode"] == "4class":
        checks.append((
            "F1 (others) (%)",
            r["test_per_class_f1"].get("others", 0),
            targets["target_f1_others"],
        ))
    print(sep)
    print(f"  {'Metric':<28} {'Score':>8}  {'Target':>8}  {'Pass':>5}")
    print(sep)
    for name, score, target in checks:
        print(f"  {name:<28} {score:>7.2f}%  {target:>7.2f}%  {'OK' if score >= target else 'FAIL':>5}")
    print(sep)
    print("  Per-class F1:")
    for cls, f1 in r["test_per_class_f1"].items():
        print(f"    {cls:<16}  {f1:>6.2f}%")
    print(sep)
    print(f"  THESIS PASS: {'YES' if r['thesis_pass'] else 'NO'}")
    print(f"{'=' * w}\n")


# ===========================================================================
# Entry point
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train chart type classifier (unified, config-driven)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "training.yaml",
        help="Path to training.yaml (default: config/training.yaml)",
    )
    parser.add_argument(
        "--override",
        action="append",
        dest="overrides",
        default=[],
        metavar="KEY=VALUE",
        help="OmegaConf dot-notation override (repeatable). "
             "Example: --override classifier.model.backbone=efficientnet_b0",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build datasets only; skip training. Useful for validating config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_manager = RunManager(
        config_path=args.config,
        cli_overrides=args.overrides,
        run_prefix="chart_classifier",
    )

    cfg: DictConfig = run_manager.config
    tracking_cfg = OmegaConf.select(cfg, "run_management", default=None)
    backend = tracking_cfg.tracking_backend if tracking_cfg else "json"

    tracker = ExperimentTracker(
        backend=backend,
        project="chart_analysis_ai_v3",
        run_name=run_manager.run_name,
        config=OmegaConf.to_container(cfg.classifier, resolve=True),
        log_dir=run_manager.run_dir / "logs",
        tags=["classifier", cfg.classifier.mode, cfg.classifier.model.backbone],
    )

    logger.info(
        f"Run started | run={run_manager.run_name} | "
        f"mode={cfg.classifier.mode} | backbone={cfg.classifier.model.backbone}"
    )

    try:
        results = train(cfg, run_manager, tracker, dry_run=args.dry_run)
        run_manager.finalize(
            metrics={
                "test_accuracy": results.get("test_accuracy"),
                "test_macro_f1": results.get("test_macro_f1"),
                "thesis_pass": results.get("thesis_pass"),
            }
        )
    except Exception as exc:
        logger.error(f"Training failed | error={exc}")
        run_manager.finalize(metrics={"error": str(exc)}, status="failed")
        tracker.finish()
        raise

    logger.info(f"Run complete | run={run_manager.run_name}")


if __name__ == "__main__":
    main()
