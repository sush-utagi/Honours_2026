"""
MS COCO Dataset Downloader & Train/Val/Test Splitter
=====================================================
Downloads the MS COCO 2017 dataset and produces consistent
70 / 15 / 15  train / val / test splits for TWO tasks:

  1. Object Detection  – bounding boxes + category labels (80 classes)
                         ground truth for ResNet fine-tuning & custom CNN
  2. Image Captioning  – 5 human-written captions per image

The SAME image split is used for both tasks so every image in
your train folder has both a detection annotation AND a caption.

Important note on test2017
──────────────────────────
COCO never released ground-truth annotations for test2017 (they
are held for the benchmark evaluation server). This script therefore
builds the detection AND caption pools from train2017 + val2017 only
(~123k fully-annotated images). test2017 images (~41k) are downloaded
but kept as a separate raw folder in case you need unlabelled images.

Output layout
─────────────
  coco_dataset/
  ├── zips/                          raw downloaded zip files
  ├── annotations/                   original extracted COCO annotations
  ├── train2017/  val2017/           original extracted images
  ├── test2017/                      unlabelled (no ground truth)
  └── split/
      ├── images/
      │   ├── train/                 symlinks → original images
      │   ├── val/
      │   └── test/
      ├── annotations/
      │   ├── instances_train.json   detection ground truth  ← ResNet / CNN
      │   ├── instances_val.json
      │   ├── instances_test.json
      │   ├── captions_train.json    captioning ground truth
      │   ├── captions_val.json
      │   └── captions_test.json
      ├── image_id_lists/
      │   ├── train.txt
      │   ├── val.txt
      │   └── test.txt
      └── split_summary.json

Usage:
    python coco_setup.py                  # full download + split
    python coco_setup.py --no-download    # split only (files already on disk)
    python coco_setup.py --copy-images    # copy instead of symlink
    python coco_setup.py --help           # all CLI options
"""

import json
import os
import shutil
import random
import zipfile
import argparse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

# Load environment variables (prefer project root .env, then local)
ROOT_ENV = Path(__file__).resolve().parents[1] / ".env"
LOCAL_ENV = Path(__file__).resolve().parent / ".env"
for env_path in (ROOT_ENV, LOCAL_ENV):
    if env_path.exists():
        load_dotenv(env_path, override=False)
# Finally, load any already-present environment variables
load_dotenv(find_dotenv(), override=False)


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    try:
        return int(val) if val is not None else default
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    try:
        return float(val) if val is not None else default
    except ValueError:
        return default


def load_config_from_env() -> dict:
    return {
        "output_dir": os.getenv("COCO_OUTPUT_DIR", "./coco_dataset"),
        "train_ratio": _env_float("COCO_TRAIN_RATIO", 0.70),
        "val_ratio":   _env_float("COCO_VAL_RATIO", 0.15),
        "test_ratio":  _env_float("COCO_TEST_RATIO", 0.15),
        "seed":        _env_int("COCO_SEED", 42),
        "download_images": _env_bool("COCO_DOWNLOAD_IMAGES", True),
        "skip_existing":   _env_bool("COCO_SKIP_EXISTING", True),
        "max_retries":     _env_int("COCO_MAX_RETRIES", 3),
    }


def load_urls_from_env() -> dict:
    return {
        "images": {
            "train2017": os.getenv("COCO_TRAIN_IMAGES_URL", "http://images.cocodataset.org/zips/train2017.zip"),
            "val2017":   os.getenv("COCO_VAL_IMAGES_URL",   "http://images.cocodataset.org/zips/val2017.zip"),
            "test2017":  os.getenv("COCO_TEST_IMAGES_URL",  "http://images.cocodataset.org/zips/test2017.zip"),
        },
        "annotations": {
            "trainval":  os.getenv("COCO_TRAINVAL_ANN_URL", "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"),
            "test_info": os.getenv("COCO_TESTINFO_ANN_URL", "http://images.cocodataset.org/annotations/image_info_test2017.zip"),
        },
    }


CONFIG = load_config_from_env()
COCO_URLS = load_urls_from_env()


# ─────────────────────────────────────────────────────────────────
#  Download helpers
# ─────────────────────────────────────────────────────────────────

def sizeof_fmt(num: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


def download_file(url: str, dest: Path, skip_existing: bool = True,
                  max_retries: int = 3) -> Path:
    """Download *url* to *dest* with a live progress bar and retry logic."""
    if skip_existing and dest.exists():
        print(f"  [skip] {dest.name} already exists.")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  Downloading {dest.name}  (attempt {attempt}/{max_retries}) …")
            with urllib.request.urlopen(url) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                done = 0
                with open(tmp, "wb") as fh:
                    while chunk := resp.read(1 << 20):   # 1 MB chunks
                        fh.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = done / total * 100
                            bar = "#" * int(pct / 2)
                            print(
                                f"\r  [{bar:<50}] {pct:5.1f}%  "
                                f"{sizeof_fmt(done)} / {sizeof_fmt(total)}",
                                end="", flush=True,
                            )
            print()
            tmp.rename(dest)
            return dest
        except Exception as exc:
            print(f"\n  ✗ Attempt {attempt} failed: {exc}")
            if tmp.exists():
                tmp.unlink()
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to download {url} after {max_retries} attempts."
                ) from exc

    return dest


def extract_zip(zip_path: Path, dest_dir: Path, skip_existing: bool = True) -> None:
    """Extract *zip_path* into *dest_dir*, skip if already extracted."""
    flag = dest_dir / f".extracted_{zip_path.stem}"
    if skip_existing and flag.exists():
        print(f"  [skip] {zip_path.name} already extracted.")
        return

    print(f"  Extracting {zip_path.name} …")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        for i, member in enumerate(members, 1):
            zf.extract(member, dest_dir)
            if i % 5000 == 0:
                print(f"    … {i:,} / {len(members):,} files", flush=True)
    flag.touch()
    print(f"  ✓ Extracted {len(members):,} files.")


# ─────────────────────────────────────────────────────────────────
#  Step 1 – Download all required files
# ─────────────────────────────────────────────────────────────────

def step_download(cfg: dict, root: Path) -> None:
    zips = root / "zips"
    zips.mkdir(parents=True, exist_ok=True)

    # ── Annotation zips ──────────────────────────────────────────
    print("\n=== Downloading annotations ===")

    # trainval: instances + captions + keypoints for train2017 & val2017
    tv_zip = zips / "annotations_trainval2017.zip"
    download_file(COCO_URLS["annotations"]["trainval"], tv_zip,
                  cfg["skip_existing"], cfg["max_retries"])
    extract_zip(tv_zip, root, cfg["skip_existing"])

    # test2017 image info (metadata only — no labels)
    ti_zip = zips / "image_info_test2017.zip"
    download_file(COCO_URLS["annotations"]["test_info"], ti_zip,
                  cfg["skip_existing"], cfg["max_retries"])
    extract_zip(ti_zip, root, cfg["skip_existing"])

    # ── Image zips ───────────────────────────────────────────────
    if cfg["download_images"]:
        print("\n=== Downloading images ===")
        for split in ("train2017", "val2017", "test2017"):
            z = zips / f"{split}.zip"
            download_file(COCO_URLS["images"][split], z,
                          cfg["skip_existing"], cfg["max_retries"])
            extract_zip(z, root, cfg["skip_existing"])


# ─────────────────────────────────────────────────────────────────
#  Step 2 – Build the combined, fully-annotated image pool
# ─────────────────────────────────────────────────────────────────

def load_json(path: Path, label: str) -> dict:
    print(f"  Loading {label} …")
    with open(path) as fh:
        return json.load(fh)


def build_annotated_pool(root: Path) -> dict:
    """
    Merge train2017 + val2017 instances AND captions into one pool.

    Only images present in BOTH the instances file AND the captions
    file are kept — guaranteeing every image in the final split has:
      • bounding-box detection labels  (ground truth for ResNet / CNN)
      • 5 human-written captions       (ground truth for captioning)

    Returns
    -------
    dict with keys:
        info, licenses, categories   – COCO metadata
        images                       – list of image dicts
        instances                    – list of bbox annotation dicts
        captions                     – list of caption annotation dicts
    """
    ann_dir = root / "annotations"

    all_images:    dict[int, dict] = {}
    all_instances: list[dict]      = []
    all_captions:  list[dict]      = []
    meta = {"info": {}, "licenses": [], "categories": []}

    instance_ids: set[int] = set()
    caption_ids:  set[int] = set()

    for split in ("train2017", "val2017"):

        # ── instances (bounding boxes + category labels) ────────
        inst_path = ann_dir / f"instances_{split}.json"
        if not inst_path.exists():
            print(f"  [warn] {inst_path.name} not found – skipping.")
        else:
            inst = load_json(inst_path, inst_path.name)
            if not meta["info"]:
                meta["info"]       = inst.get("info", {})
                meta["licenses"]   = inst.get("licenses", [])
                meta["categories"] = inst.get("categories", [])
            for img in inst.get("images", []):
                all_images[img["id"]] = img
                instance_ids.add(img["id"])
            all_instances.extend(inst.get("annotations", []))

        # ── captions ────────────────────────────────────────────
        cap_path = ann_dir / f"captions_{split}.json"
        if not cap_path.exists():
            print(f"  [warn] {cap_path.name} not found – skipping.")
        else:
            cap = load_json(cap_path, cap_path.name)
            for img in cap.get("images", []):
                caption_ids.add(img["id"])
            all_captions.extend(cap.get("annotations", []))

    # Intersect: keep only images that have BOTH annotation types
    valid_ids = instance_ids & caption_ids
    missing   = (instance_ids | caption_ids) - valid_ids
    if missing:
        print(f"  [info] Dropped {len(missing):,} images missing one annotation type.")

    images    = [img for img_id, img in all_images.items() if img_id in valid_ids]
    instances = [a   for a in all_instances if a["image_id"] in valid_ids]
    captions  = [a   for a in all_captions  if a["image_id"] in valid_ids]

    n_img = len(images)
    print(f"\n  ┌─────────────────────────────────────────────────")
    print(f"  │ Pool: {n_img:,} fully-annotated images")
    print(f"  │   Detection annotations : {len(instances):,} bounding boxes")
    print(f"  │   Captions              : {len(captions):,}  "
          f"(avg {len(captions)/n_img:.1f} per image)")
    print(f"  └─────────────────────────────────────────────────")

    return {**meta, "images": images, "instances": instances, "captions": captions}


# ─────────────────────────────────────────────────────────────────
#  Step 3 – Partition images
# ─────────────────────────────────────────────────────────────────

def split_images(pool: dict, train_r: float, val_r: float, test_r: float,
                 seed: int) -> dict[str, list]:
    """Randomly shuffle and partition pool images into train / val / test."""
    total = train_r + val_r + test_r
    assert abs(total - 1.0) < 1e-6, f"Ratios must sum to 1.0 (got {total:.4f})"

    images = pool["images"][:]
    random.seed(seed)
    random.shuffle(images)

    n       = len(images)
    n_train = round(n * train_r)
    n_val   = round(n * val_r)
    n_test  = n - n_train - n_val   # remainder avoids rounding drift

    splits: dict[str, list] = {
        "train": images[:n_train],
        "val":   images[n_train : n_train + n_val],
        "test":  images[n_train + n_val :],
    }

    print(f"\n  {'Split':<6}  {'Images':>8}  {'%':>6}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*6}")
    for name, imgs in splits.items():
        print(f"  {name:<6}  {len(imgs):>8,}  {len(imgs)/n*100:>5.1f}%")
    print(f"  {'─'*6}  {'─'*8}  {'─'*6}")
    print(f"  {'total':<6}  {n:>8,}  100.0%")

    return splits


# ─────────────────────────────────────────────────────────────────
#  Step 4 – Write all outputs
# ─────────────────────────────────────────────────────────────────

def _filter_anns(anns: list[dict], ids: set[int]) -> list[dict]:
    return [a for a in anns if a["image_id"] in ids]


def write_detection_annotations(splits: dict, pool: dict, ann_out: Path) -> None:
    """
    Write instances_{train,val,test}.json in standard COCO format.

    Each file contains:
      categories  – 80 COCO object classes
      images      – image metadata for this split
      annotations – bounding boxes with category_id, bbox [x,y,w,h],
                    area, iscrowd  (everything your ResNet / CNN needs)
    """
    print("\n  Detection annotation files (instances_*.json):")
    for split_name, images in splits.items():
        ids  = {img["id"] for img in images}
        anns = _filter_anns(pool["instances"], ids)
        doc  = {
            "info":        pool["info"],
            "licenses":    pool["licenses"],
            "categories":  pool["categories"],
            "images":      images,
            "annotations": anns,
        }
        path = ann_out / f"instances_{split_name}.json"
        with open(path, "w") as fh:
            json.dump(doc, fh)
        print(f"    {path.name:<30}  {len(images):>7,} images  "
              f"{len(anns):>8,} bbox annotations")


def write_caption_annotations(splits: dict, pool: dict, ann_out: Path) -> None:
    """
    Write captions_{train,val,test}.json in standard COCO format.

    Each annotation dict contains:
      id         – unique annotation id
      image_id   – links to the image
      caption    – human-written sentence describing the image
    (~5 captions per image on average)
    """
    print("\n  Caption annotation files (captions_*.json):")
    for split_name, images in splits.items():
        ids  = {img["id"] for img in images}
        caps = _filter_anns(pool["captions"], ids)
        doc  = {
            "info":        pool["info"],
            "licenses":    pool["licenses"],
            "images":      images,
            "annotations": caps,
        }
        path = ann_out / f"captions_{split_name}.json"
        with open(path, "w") as fh:
            json.dump(doc, fh)
        print(f"    {path.name:<30}  {len(images):>7,} images  "
              f"{len(caps):>8,} captions  "
              f"(avg {len(caps)/len(images):.1f}/img)")


def organise_images(splits: dict, root: Path, out_dir: Path,
                    use_symlinks: bool = True) -> None:
    """
    Populate out_dir/images/{train,val,test}/ using symlinks (default,
    zero extra disk space) or hard copies.
    Only train2017 + val2017 images are mapped — test2017 has no labels.
    """
    img_map: dict[str, Path] = {}
    for src_dir in (root / "train2017", root / "val2017"):
        if src_dir.is_dir():
            for p in src_dir.iterdir():
                if p.is_file():
                    img_map[p.name] = p

    if not img_map:
        print("  [warn] No extracted image folders found. "
              "Run without --no-download, or extract manually first.")
        return

    verb = "Symlinking" if use_symlinks else "Copying"
    print(f"\n  {verb} images into split directories …")
    for split_name, images in splits.items():
        dest_dir = out_dir / "images" / split_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        missing = 0
        for img in images:
            fname = img["file_name"].split("/")[-1]
            src   = img_map.get(fname)
            if src is None:
                missing += 1
                continue
            dest = dest_dir / fname
            if dest.exists() or dest.is_symlink():
                continue
            if use_symlinks:
                dest.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dest)
        status = f"  ⚠  {missing:,} not found on disk" if missing else ""
        print(f"    {split_name:<5}  {len(images) - missing:>7,} images{status}")
    print("  ✓ Image directories ready.")


def write_id_lists(splits: dict, out_dir: Path) -> None:
    """Write plain-text image ID files — handy for custom data loaders."""
    lists_dir = out_dir / "image_id_lists"
    lists_dir.mkdir(parents=True, exist_ok=True)
    for name, images in splits.items():
        path = lists_dir / f"{name}.txt"
        with open(path, "w") as fh:
            fh.write("\n".join(str(img["id"]) for img in images) + "\n")
    print("  ✓ Image ID lists written.")


def write_summary(splits: dict, pool: dict, cfg: dict, out_dir: Path) -> None:
    """Write a human-readable JSON summary of the whole operation."""
    summary: dict = {
        "config": cfg,
        "pool": {
            "total_images":                len(pool["images"]),
            "total_detection_annotations": len(pool["instances"]),
            "total_captions":              len(pool["captions"]),
        },
        "splits": {},
    }
    for name, images in splits.items():
        ids = {img["id"] for img in images}
        summary["splits"][name] = {
            "images":                len(images),
            "detection_annotations": sum(1 for a in pool["instances"]
                                         if a["image_id"] in ids),
            "captions":              sum(1 for a in pool["captions"]
                                         if a["image_id"] in ids),
        }
    path = out_dir / "split_summary.json"
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  ✓ Summary written → {path.name}")


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download MS COCO 2017 and split for detection + captioning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir",  default=CONFIG["output_dir"],
                        help="Root directory for all dataset files.")
    parser.add_argument("--train-ratio", type=float, default=CONFIG["train_ratio"])
    parser.add_argument("--val-ratio",   type=float, default=CONFIG["val_ratio"])
    parser.add_argument("--test-ratio",  type=float, default=CONFIG["test_ratio"])
    parser.add_argument("--seed",        type=int,   default=CONFIG["seed"])
    parser.add_argument("--no-download", action="store_true",
                        help="Skip downloads; split already-extracted files only.")
    parser.add_argument("--copy-images", action="store_true",
                        help="Copy images instead of symlinking (uses more disk).")
    args = parser.parse_args()

    cfg = {**CONFIG,
           "output_dir":  args.output_dir,
           "train_ratio": args.train_ratio,
           "val_ratio":   args.val_ratio,
           "test_ratio":  args.test_ratio,
           "seed":        args.seed}

    root    = Path(cfg["output_dir"]).resolve()
    out_dir = root / "split"
    root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*62}")
    print("  MS COCO 2017  ·  Object Detection + Image Captioning")
    print(f"  Output  : {root}")
    print(f"  Split   : {cfg['train_ratio']:.0%} train / "
          f"{cfg['val_ratio']:.0%} val / "
          f"{cfg['test_ratio']:.0%} test")
    print(f"  Seed    : {cfg['seed']}")
    print(f"{'='*62}")

    # 1 ── Download
    if not args.no_download:
        print("\nSTEP 1/4 – Download")
        step_download(cfg, root)
    else:
        print("\nSTEP 1/4 – Download  [skipped via --no-download]")

    # 2 ── Build pool
    print("\nSTEP 2/4 – Build fully-annotated image pool")
    pool = build_annotated_pool(root)

    # 3 ── Split
    print("\nSTEP 3/4 – Partition images")
    splits = split_images(
        pool,
        cfg["train_ratio"], cfg["val_ratio"], cfg["test_ratio"], cfg["seed"],
    )

    # 4 ── Write outputs
    print("\nSTEP 4/4 – Write outputs")
    ann_out = out_dir / "annotations"
    ann_out.mkdir(parents=True, exist_ok=True)

    write_detection_annotations(splits, pool, ann_out)
    write_caption_annotations(splits, pool, ann_out)
    organise_images(splits, root, out_dir, use_symlinks=not args.copy_images)
    write_id_lists(splits, out_dir)
    write_summary(splits, pool, cfg, out_dir)

    # ── Done
    print(f"\n{'='*62}")
    print("  All done! Your dataset is ready at:")
    print(f"    {out_dir}/")
    print("    ├── annotations/")
    print("    │   ├── instances_train.json  ← bbox labels (ResNet / CNN)")
    print("    │   ├── instances_val.json")
    print("    │   ├── instances_test.json")
    print("    │   ├── captions_train.json   ← text captions (captioning)")
    print("    │   ├── captions_val.json")
    print("    │   └── captions_test.json")
    print("    ├── images/")
    print("    │   ├── train/")
    print("    │   ├── val/")
    print("    │   └── test/")
    print("    ├── image_id_lists/")
    print("    │   ├── train.txt")
    print("    │   ├── val.txt")
    print("    │   └── test.txt")
    print("    └── split_summary.json")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
