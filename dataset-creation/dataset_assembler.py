"""Multi-class COCO dataset assembler with synthetic splicing.

Builds train/val/test classification datasets over all 80 MS COCO categories,
while allowing synthetic images to be injected into train/val for any target
class. Images are resized to 512x512 and normalized to [-1, 1].

Classes are derived from official COCO category metadata and mapped to a
contiguous index range [0, 79].
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset

try:  # optional reporting dependency
    import pandas as pd
except ImportError:  # pragma: no cover - handled at runtime
    pd = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _pil_to_tensor_512(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((512, 512), resample=Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # HWC -> CHW
    tensor = tensor * 2.0 - 1.0  # normalize to [-1, 1]
    return tensor


@dataclass
class Sample:
    path: Path
    label: int
    synthetic: bool


class MultiClassDataset(Dataset):
    def __init__(self, samples: Sequence[Sample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, bool]:
        sample = self.samples[idx]
        img = Image.open(sample.path)
        tensor = _pil_to_tensor_512(img)
        return tensor, sample.label, sample.synthetic


class HybridDatasetAssembler:
    """Assemble COCO multiclass datasets with optional synthetic injection."""

    def __init__(self, coco_root: str = "coco_dataset/split", seed: int = 42):
        self.coco_root = Path(coco_root)
        self.seed = seed
        random.seed(seed)
        self.cat_id_to_idx: Dict[int, int] = {}
        self.idx_to_name: List[str] = []

    # ------------------------------------------------------------------ public
    def assemble(self, synthetic_dir_name: Optional[str], target_class_name: Optional[str]) -> Dict[str, MultiClassDataset]:
        """Create train/val/test datasets.

        Args:
            synthetic_dir_name: subfolder under data-generation-outputs/diffusion containing synthetic images.
                                If None/empty, no synthetic data is injected.
            target_class_name: COCO category name to assign to synthetic images.
                               Required when synthetic_dir_name is provided.
        """

        splits: Dict[str, List[Sample]] = {"train": [], "val": [], "test": []}

        # Build category mapping from train annotations (authoritative)
        train_instances = self.coco_root / "annotations" / "instances_train.json"
        if not train_instances.exists():
            raise FileNotFoundError(f"Missing COCO annotations: {train_instances}")
        coco_train = COCO(str(train_instances))
        self._build_category_mapping(coco_train)

        # Load real COCO data for each split
        for split in ["train", "val", "test"]:
            instances_path = self.coco_root / "annotations" / f"instances_{split}.json"
            images_dir = self.coco_root / "images" / split
            if not instances_path.exists():
                raise FileNotFoundError(f"Missing COCO annotations: {instances_path}")
            coco = COCO(str(instances_path))
            samples = self._collect_coco_samples(coco, images_dir)
            splits[split].extend(samples)

        # Synthetic injection (train/val only)
        if synthetic_dir_name:
            if target_class_name is None:
                raise ValueError("target_class_name must be provided when using synthetic_dir_name")
            target_idx = self._target_label_index(target_class_name)
            synth_train, synth_val = self._load_synthetic(synthetic_dir_name, target_idx)

            splits["train"].extend(synth_train)
            splits["val"].extend(synth_val)

        # Shuffle to avoid real/synthetic ordering artifacts
        random.shuffle(splits["train"])
        random.shuffle(splits["val"])

        datasets = {k: MultiClassDataset(v) for k, v in splits.items()}
        self._log_distribution(datasets, synthetic_label=target_class_name if synthetic_dir_name else None)
        return datasets

    # ---------------------------------------------------------------- utilities
    def _build_category_mapping(self, coco: COCO) -> None:
        cats = coco.loadCats(coco.getCatIds())
        cats_sorted = sorted(cats, key=lambda c: c["id"])
        self.cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats_sorted)}
        self.idx_to_name = [c["name"] for c in cats_sorted]
        if len(self.idx_to_name) != 80:
            raise ValueError(f"Expected 80 COCO classes, found {len(self.idx_to_name)}")

    def _collect_coco_samples(self, coco: COCO, images_dir: Path) -> List[Sample]:
        samples: List[Sample] = []
        for img_id, img_info in coco.imgs.items():
            file_name = img_info["file_name"]
            path = images_dir / file_name
            if not path.exists():
                continue

            ann_ids = coco.getAnnIds(imgIds=[img_id])
            anns = coco.loadAnns(ann_ids)
            if not anns:
                continue

            primary_cat = self._primary_category(anns)
            label_idx = self.cat_id_to_idx.get(primary_cat)
            if label_idx is None:
                continue
            samples.append(Sample(path=path, label=label_idx, synthetic=False))
        return samples

    def _primary_category(self, anns: List[dict]) -> int:
        """Select category id of the largest-area annotation."""
        def area(a: dict) -> float:
            if "area" in a:
                return float(a["area"])
            # bbox = [x,y,w,h]
            bbox = a.get("bbox", [0, 0, 0, 0])
            return float(bbox[2] * bbox[3])

        return max(anns, key=area)["category_id"]

    def _target_label_index(self, target_class_name: str) -> int:
        if not self.idx_to_name:
            raise RuntimeError("Category mapping not initialized")
        try:
            return self.idx_to_name.index(target_class_name)
        except ValueError as exc:
            raise ValueError(f"target_class_name '{target_class_name}' not found in COCO categories") from exc

    def _load_synthetic(self, synthetic_dir_name: str, target_idx: int) -> Tuple[List[Sample], List[Sample]]:
        synth_root = Path("data-generation-outputs") / "diffusion" / synthetic_dir_name
        if not synth_root.exists():
            raise FileNotFoundError(f"Synthetic directory not found: {synth_root}")

        image_paths = [
            p for p in synth_root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        if not image_paths:
            raise ValueError(f"No images found under {synth_root}")

        random.shuffle(image_paths)
        split_idx = int(0.8 * len(image_paths))
        train_paths = image_paths[:split_idx]
        val_paths = image_paths[split_idx:]

        train_samples = [Sample(path=p, label=target_idx, synthetic=True) for p in train_paths]
        val_samples = [Sample(path=p, label=target_idx, synthetic=True) for p in val_paths]
        return train_samples, val_samples

    def _log_distribution(self, datasets: Dict[str, MultiClassDataset], synthetic_label: Optional[str]) -> None:
        print("\nFinal dataset distribution (real vs synthetic per class)")
        for split, ds in datasets.items():
            totals = {i: {"real": 0, "synthetic": 0} for i in range(len(self.idx_to_name))}
            for _, label, synthetic in ds:
                if synthetic:
                    totals[label]["synthetic"] += 1
                else:
                    totals[label]["real"] += 1

            if pd is not None:
                df = pd.DataFrame.from_dict(totals, orient="index")
                df["class_name"] = [self.idx_to_name[i] for i in range(len(self.idx_to_name))]
                cols = ["class_name", "real", "synthetic"]
                df = df[cols]
                print(f"\n{split.upper()}:\n{df.to_string(index=False)}")
            else:
                print(f"\n{split.upper()}:")
                for idx in range(len(self.idx_to_name)):
                    name = self.idx_to_name[idx]
                    real = totals[idx]["real"]
                    synth = totals[idx]["synthetic"]
                    marker = " <- synthetic target" if synthetic_label == name and synth > 0 else ""
                    print(f"  {idx:02d} {name:20s} real={real:6d} synthetic={synth:6d}{marker}")
