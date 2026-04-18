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

    def __init__(
        self,
        coco_root: str = "coco_dataset/split",
        contextual_root: str = "coco_dataset/contextual_crops",
        seed: int = 42,
    ):
        # Raw COCO root (used for test split fallback) and contextual crops root (preferred for train/val)
        self.coco_root = Path(coco_root)
        self.contextual_root = Path(contextual_root)
        self.seed = seed
        random.seed(seed)
        self.cat_id_to_idx: Dict[int, int] = {}
        self.idx_to_name: List[str] = []

    # ------------------------------------------------------------------ public
    def assemble(
        self,
        synthetic_dir_name: Optional[str],
        target_class_name: Optional[str],
        load_test: bool = True,
    ) -> Dict[str, MultiClassDataset]:
        """Create train/val/test datasets.

        Args:
            synthetic_dir_name: subfolder under data-generation-outputs/diffusion containing synthetic images.
                                If None/empty, no synthetic data is injected.
            target_class_name: COCO category name to assign to synthetic images.
                                Required when synthetic_dir_name is provided.
            load_test: when False, skip loading the test split entirely (useful to keep test untouched).
        """

        splits: Dict[str, List[Sample]] = {"train": [], "val": []}
        if load_test:
            splits["test"] = []

        # Build category mapping from train annotations (authoritative)
        self._ensure_category_mapping()

        # Load real COCO data for each split
        for split in ["train", "val"] + (["test"] if load_test else []):
            try:
                instances_path, images_dir = self._paths_for_split(split)
            except FileNotFoundError as exc:
                if split == "test":
                    print(f"[skip] {exc}")
                    continue
                raise

            coco = COCO(str(instances_path))
            samples = self._collect_coco_samples(coco, images_dir)
            splits[split].extend(samples)
            print(f"[data] loaded {len(samples):,} real samples for {split}")

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


    def get_image_paths_by_class(
        self,
        split: str, # 'train', 'val', 'test'
        target_classes: Optional[Sequence[str]] = None, # only collect paths for these class names
        max_samples: Optional[int] = None,
    ) -> Dict[str, List[Path]]:
        """
        Group image paths by their primary category for a given split.
            
        Returns:
            Dict mapping class name -> list of absolute Paths to images.
        """
        self._ensure_category_mapping()
        
        try:
            instances_path, images_dir = self._paths_for_split(split)
        except FileNotFoundError:
            return {}

        coco = COCO(str(instances_path))
        
        target_indices = None
        if target_classes is not None:
            target_indices = set()
            for cls_name in target_classes:
                try:
                    idx = self._target_label_index(cls_name)
                    target_indices.add(idx)
                except ValueError:
                    pass

        paths_by_class: Dict[int, List[Path]] = {}
        for img_id, img_info in coco.imgs.items():
            path = images_dir / img_info["file_name"]
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
                
            if target_indices is not None and label_idx not in target_indices:
                continue
                
            paths_by_class.setdefault(label_idx, []).append(path)

        result: Dict[str, List[Path]] = {}
        for idx, paths in paths_by_class.items():
            class_name = self.idx_to_name[idx]
            if max_samples is not None and len(paths) > max_samples:
                rng = random.Random(self.seed)
                paths = rng.sample(paths, max_samples)
            result[class_name] = sorted(paths)
            
        return result

    def get_image_to_label_mapping(self, split: str) -> Dict[str, int]:
        """
        Returns mapping of image file name to the primary class ID 
        (contiguous label index) for the given split.
        """
        self._ensure_category_mapping()
        
        try:
            instances_path, _ = self._paths_for_split(split)
        except FileNotFoundError:
            return {}

        coco = COCO(str(instances_path))
        mapping: Dict[str, int] = {}
        for img_id, img_info in coco.imgs.items():
            ann_ids = coco.getAnnIds(imgIds=[img_id])
            anns = coco.loadAnns(ann_ids)
            if not anns:
                continue
            primary_cat = self._primary_category(anns)
            label_idx = self.cat_id_to_idx.get(primary_cat)
            if label_idx is not None:
                mapping[img_info["file_name"]] = label_idx
                
        return mapping

    def _ensure_category_mapping(self) -> None:
        """Ensures that the category mappings (name to id and vice versa) are loaded."""
        if not self.idx_to_name:
            train_instances, _ = self._paths_for_split("train")
            coco_train = COCO(str(train_instances))
            self._build_category_mapping(coco_train)

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

    def _paths_for_split(self, split: str) -> Tuple[Path, Path]:
        """Return (annotations_path, images_dir) preferring contextual crops for train/val."""

        # Prefer contextual crops when available.
        contextual_ann = self.contextual_root / "annotations" / f"single_instances_{split}.json"
        contextual_imgs = self.contextual_root / "images" / split

        if split in {"train", "val"}:
            if not contextual_ann.exists():
                raise FileNotFoundError(
                    f"Missing contextual crop annotations for '{split}' at {contextual_ann}. "
                    "Run dataset-creation/preprocess_coco.py to generate contextual_crops."
                )
            return contextual_ann, contextual_imgs

        # For test (or other splits), prefer contextual if present, else fall back to COCO.
        if contextual_ann.exists() and contextual_imgs.exists():
            return contextual_ann, contextual_imgs

        ann = self.coco_root / "annotations" / f"instances_{split}.json"
        imgs = self.coco_root / "images" / split
        if not ann.exists():
            raise FileNotFoundError(f"Missing COCO annotations for split '{split}': {ann}")
        return ann, imgs

    def _log_distribution(self, datasets: Dict[str, MultiClassDataset], synthetic_label: Optional[str]) -> None:
        pass
        # print("\nFinal dataset distribution (real vs synthetic per class)")
        # for split, ds in datasets.items():
        #     totals = {i: {"real": 0, "synthetic": 0} for i in range(len(self.idx_to_name))}
        #     for _, label, synthetic in ds:
        #         if synthetic:
        #             totals[label]["synthetic"] += 1
        #         else:
        #             totals[label]["real"] += 1

        #     if pd is not None:
        #         df = pd.DataFrame.from_dict(totals, orient="index")
        #         df["class_name"] = [self.idx_to_name[i] for i in range(len(self.idx_to_name))]
        #         cols = ["class_name", "real", "synthetic"]
        #         df = df[cols]
        #         print(f"\n{split.upper()}:\n{df.to_string(index=False)}")
        #     else:
        #         print(f"\n{split.upper()}:")
        #         for idx in range(len(self.idx_to_name)):
        #             name = self.idx_to_name[idx]
        #             real = totals[idx]["real"]
        #             synth = totals[idx]["synthetic"]
        #             marker = " <- synthetic target" if synthetic_label == name and synth > 0 else ""
        #             print(f"  {idx:02d} {name:20s} real={real:6d} synthetic={synth:6d}{marker}")
