"""Simple ResNet classifier training & evaluation utilities.

This module provides:
 - A light ResNet-18 style backbone implemented in pure PyTorch.
 - ImageFolder-style dataset loading without torchvision.
 - Training / validation loops with accuracy tracking.
 - Precision–Recall curve plotting.
 - Grad-CAM based feature-importance heatmaps for image inputs.

The design aims to stay dependency-light while remaining easy to extend later.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

# Use headless backend for notebook/CLI environments without displays
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm import tqdm


###############################################################################
# Data utilities
###############################################################################


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_transform(image_size: int = 224, augment: bool = False) -> Callable[[Image.Image], torch.Tensor]:
    """Create a basic PIL -> normalized tensor transform.

    Args:
        image_size: Final square size for resizing.
        augment: Apply random horizontal flip when True.
    """

    def _transform(img: Image.Image) -> torch.Tensor:
        img = img.convert("RGB")
        img = img.resize((image_size, image_size), resample=Image.BILINEAR)
        if augment and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = (arr - MEAN) / STD
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW
        return tensor

    return _transform


class SimpleImageFolder(Dataset):
    """Lightweight alternative to torchvision.datasets.ImageFolder."""

    def __init__(self, root: str, transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
                 extensions: Sequence[str] = IMAGE_EXTENSIONS) -> None:
        self.root = root
        self.transform = transform
        self.extensions = tuple(ext.lower() for ext in extensions)

        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.samples: List[Tuple[str, int]] = []

        for cls_name in sorted(os.listdir(root)):
            cls_path = os.path.join(root, cls_name)
            if not os.path.isdir(cls_path):
                continue
            idx = len(self.classes)
            self.classes.append(cls_name)
            self.class_to_idx[cls_name] = idx
            for fname in sorted(os.listdir(cls_path)):
                if fname.lower().endswith(self.extensions):
                    self.samples.append((os.path.join(cls_path, fname), idx))

        if not self.samples:
            raise ValueError(f"No image files with extensions {self.extensions} found under {root}")

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class TransformSubset(Subset):
    """Subset wrapper that applies a transform to the images."""

    def __init__(self, subset: Subset, transform: Callable[[Image.Image], torch.Tensor]):
        super().__init__(subset.dataset, subset.indices)
        self.transform = transform

    def __getitem__(self, idx: int):
        # Subset maps idx -> original dataset index
        image, label = self.dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    val_split: float = 0.2,
    augment: bool = True,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Create train/validation DataLoaders from an ImageFolder-style directory."""

    base_ds = SimpleImageFolder(data_dir, transform=None)
    n_val = max(1, int(len(base_ds) * val_split))
    n_train = len(base_ds) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(base_ds, [n_train, n_val], generator=generator)

    train_ds = TransformSubset(train_subset, build_transform(image_size, augment=augment))
    val_ds = TransformSubset(val_subset, build_transform(image_size, augment=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, base_ds.classes


###############################################################################
# Model definition (ResNet-18 style)
###############################################################################


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block: Callable, layers: Sequence[int], num_classes: int):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: Callable, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [block(self.in_planes, planes, stride)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_resnet18(num_classes: int) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


###############################################################################
# Training / evaluation
###############################################################################


@dataclass
class TrainHistory:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer,
                   device: torch.device, progress_desc: Optional[str] = None, show_progress: bool = False) -> Tuple[float, float]:
    model.train()
    running_loss, running_acc, n_batches = 0.0, 0.0, 0
    loop = tqdm(loader, desc=progress_desc, leave=False) if show_progress else loader
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits, labels)
        n_batches += 1

    return running_loss / n_batches, running_acc / n_batches


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
             progress_desc: Optional[str] = None, show_progress: bool = False) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss, running_acc, n_batches = 0.0, 0.0, 0
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    loop = tqdm(loader, desc=progress_desc, leave=False) if show_progress else loader
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item()
        running_acc += accuracy_from_logits(logits, labels)
        n_batches += 1
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits_concat = torch.cat(all_logits, dim=0)
    labels_concat = torch.cat(all_labels, dim=0)
    probs = F.softmax(logits_concat, dim=1).numpy()
    labels_np = labels_concat.numpy()
    return running_loss / n_batches, running_acc / n_batches, labels_np, probs


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    show_progress: bool = False,
) -> TrainHistory:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = TrainHistory()
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            progress_desc=f"Train {epoch+1}/{epochs}",
            show_progress=show_progress,
        )
        val_loss, val_acc, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
            progress_desc=f"Val {epoch+1}/{epochs}",
            show_progress=show_progress,
        )

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.train_acc.append(train_acc)
        history.val_acc.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

    return history


###############################################################################
# Metrics & visualizations
###############################################################################


def plot_training_curves(history: TrainHistory, save_path: Optional[str] = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.train_loss, label="train")
    axes[0].plot(history.val_loss, label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.train_acc, label="train")
    axes[1].plot(history.val_acc, label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()
    plt.close(fig)


def _precision_recall_points(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]
    tp, fp = 0, 0
    precisions, recalls = [], []
    total_pos = float(y_true.sum())
    if total_pos == 0:
        return np.array([1.0]), np.array([0.0])
    for label in y_true_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / total_pos)
    # start at recall 0 with precision 1
    precisions.insert(0, 1.0)
    recalls.insert(0, 0.0)
    return np.array(precisions), np.array(recalls)


def _average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    # trapezoidal area under PR curve
    return float(np.trapz(precision, recall))


def plot_precision_recall_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Sequence[str],
    save_path: Optional[str] = None,
) -> Dict[str, float]:
    """Plot per-class precision–recall curves and return average precision per class."""

    num_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(7, 5))
    average_precisions: Dict[str, float] = {}

    for cls_idx, cls_name in enumerate(class_names):
        y_true_c = (y_true == cls_idx).astype(int)
        precision, recall = _precision_recall_points(y_true_c, y_prob[:, cls_idx])
        ap = _average_precision(precision, recall)
        average_precisions[cls_name] = ap
        ax.plot(recall, precision, label=f"{cls_name} (AP={ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    else:
        plt.show()
    plt.close(fig)
    return average_precisions


###############################################################################
# Grad-CAM feature importance
###############################################################################


def compute_gradcam(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int,
    device: torch.device,
    target_layer: Optional[nn.Module] = None,
) -> np.ndarray:
    """Compute a Grad-CAM heatmap for a single image tensor (C,H,W)."""

    model.eval()
    target_layer = target_layer or getattr(model, "layer4")[-1].conv2

    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

    def forward_hook(_module, _input, output):
        activations.append(output)

    def backward_hook(_module, grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)

    image = image.unsqueeze(0).to(device)
    model.zero_grad()
    scores = model(image)
    score = scores[0, target_class]
    score.backward()

    act = activations[-1].detach()[0]  # (C,H,W)
    grad = gradients[-1].detach()[0]   # (C,H,W)
    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * act).sum(dim=0))
    cam -= cam.min()
    cam /= cam.max() + 1e-6
    heatmap = cam.cpu().numpy()

    handle_fwd.remove()
    handle_bwd.remove()
    return heatmap


def save_gradcam_overlay(
    image: torch.Tensor,
    heatmap: np.ndarray,
    save_path: str,
    alpha: float = 0.4,
) -> None:
    """Overlay heatmap on the original image and save."""

    img = image.permute(1, 2, 0).cpu().numpy()
    img = (img * STD + MEAN).clip(0, 1)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img)
    ax.imshow(heatmap, cmap="jet", alpha=alpha)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def generate_feature_importance(
    model: nn.Module,
    loader: DataLoader,
    class_names: Sequence[str],
    output_dir: str,
    max_images: int = 4,
    device: Optional[torch.device] = None,
    target_layer: Optional[nn.Module] = None,
) -> List[str]:
    """Generate Grad-CAM overlays for a handful of images.

    Returns a list of saved file paths.
    """

    os.makedirs(output_dir, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved_paths: List[str] = []
    model.to(device)

    processed = 0
    for images, labels in loader:
        for i in range(len(images)):
            if processed >= max_images:
                return saved_paths
            image = images[i]
            label = int(labels[i])
            heatmap = compute_gradcam(model, image, label, device=device, target_layer=target_layer)
            fname = os.path.join(output_dir, f"gradcam_{processed}_{class_names[label]}.png")
            save_gradcam_overlay(image, heatmap, fname)
            saved_paths.append(fname)
            processed += 1
        if processed >= max_images:
            break
    return saved_paths


###############################################################################
# Convenience CLI entrypoint
###############################################################################


def run_experiment(
    data_dir: str,
    epochs: int = 5,
    batch_size: int = 32,
    image_size: int = 224,
    lr: float = 1e-3,
    output_dir: str = "runs/resnet",
    val_split: float = 0.2,
    augment: bool = True,
    num_workers: int = 0,
    seed: int = 42,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    train_loader, val_loader, class_names = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        val_split=val_split,
        augment=augment,
        num_workers=num_workers,
        seed=seed,
    )

    model = build_resnet18(num_classes=len(class_names))
    history = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr)

    # final evaluation for PR curves
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loss, val_acc, y_true, y_prob = evaluate(model.to(device), val_loader, criterion, device)
    print(f"Validation: loss={val_loss:.4f} acc={val_acc:.3f}")

    plot_training_curves(history, save_path=os.path.join(output_dir, "training_curves.png"))
    plot_precision_recall_curves(y_true, y_prob, class_names, save_path=os.path.join(output_dir, "precision_recall.png"))
    generate_feature_importance(
        model,
        val_loader,
        class_names,
        output_dir=os.path.join(output_dir, "gradcam"),
        max_images=4,
        device=device,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a simple ResNet classifier")
    parser.add_argument("data_dir", help="Path to dataset root (ImageFolder style)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="runs/resnet")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--no_augment", action="store_true", help="Disable random horizontal flip")
    args = parser.parse_args()

    run_experiment(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr=args.lr,
        output_dir=args.output_dir,
        val_split=args.val_split,
        augment=not args.no_augment,
    )
