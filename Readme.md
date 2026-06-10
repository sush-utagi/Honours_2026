# Honours 2026 — Synthetic Data Augmentation for Image Classification

## Repository Structure

```
Honours_2026/
│
├── scripts/                            # Setup, training, and analysis scripts
│   ├── coco_setup.py                   #   Download COCO 2017 & create 70/15/15 splits
│   ├── add_synthetic_to_train.py       #   Inject synthetic images into training set
│   ├── train_embeddings.sh             #   Launch textual-inversion training
│   ├── run_all_evaluations.sh          #   Evaluate all saved models in batch
│   └── ...                             #   Plotting & utility helpers
│
├── experiments/
│   │
│   ├── dataset_creation/               # ★ Data ingestion & transformation
│   │   ├── preprocess_coco.py          #   Convert COCO detections → 512×512 contextual
│   │   │                               #   crops (single-object classification images)
│   │   ├── dataset_assembler.py        #   Merge real crops + synthetic images into a
│   │   │                               #   unified hybrid training set
│   │   ├── class_distribution.py       #   Visualise per-class sample counts
│   │   └── visualize_crops.py          #   Quick visual sanity-check of crop outputs
│   │
│   ├── evaluation_module/
│   │   ├── classifier/
│   │   │   └── resnet_classifier.py    # ★ ResNet-18 implementation (pure PyTorch):
│   │   │                               #   training loop, evaluation, Grad-CAM,
│   │   │                               #   precision–recall curves
│   │   ├── captioner/
│   │   │   └── caption_cli.py          #   CLI for generating / handling captions
│   │   └── grade_images/
│   │       ├── analyse.py              #   Comprehensive synthetic-image analysis
│   │       ├── grader.py               #   CLIP-based quality grading of generated images
│   │       └── plot_*.py               #   Plotting scripts (CLIP distributions,
│   │                                   #   memorisation ratios, diversity, etc.)
│   │
│   ├── model/
│   │   ├── train_model.py              # ★ Main training entry-point (baseline vs.
│   │   │                               #   experimental configurations)
│   │   └── test_model.py               #   Standalone model evaluation / inference
│   │
│   ├── analyses/
│   │   ├── umap_baseline.py            #   UMAP embedding visualisations
│   │   └── plot_sweeps.py              #   Hyperparameter sweep plots
│   │
│   └── figures/                        # Generated figures and visualisations
│
├── data_generation_backend/            # Synthetic image generation
│   ├── diffusion_runner.py             #   Stable Diffusion pipeline (txt2img, img2img,
│   │                                   #   ControlNet); supports prompt sweeps & JSON
│   │                                   #   batch configs
│   ├── textual_inversion.py            #   Textual-inversion fine-tuning
│   ├── diffusion_model/                #   Local SD implementation (git submodule)
│   └── prompt_generation/              #   Prompt construction & shuffling utilities
│       ├── prompt_shuffler.py          #     Combinatorial prompt builder
│       ├── select_controlnet_candidates.py
│       └── select_ip_references.py     #     Reference-image selection for IP-Adapter
│
├── coco_dataset/                       # COCO 2017 data (gitignored images)
│   ├── annotations/                    #   Original COCO annotation JSONs
│   ├── contextual_crops/               #   512×512 classification crops (output of
│   │                                   #   preprocess_coco.py)
│   └── split/                          #   Train / val / test split manifests
│
├── runs/                               # Training checkpoints & Grad-CAM outputs
├── results/                            # Saved evaluation results per model
├── logs/                               # Training logs
├── data_generation_outputs/            # Raw synthetic image outputs
│
├── gui/
│   └── main.py                         # Tkinter GUI for interactive generation
│
├── thesis/                             # LaTeX source for the dissertation
├── requirements.txt                    # Python dependencies
└── .env                                # Environment config (model IDs, cache paths)
```

### Key files at a glance

| File | Role |
|------|------|
| `experiments/dataset_creation/preprocess_coco.py` | Converts COCO bounding-box annotations into 512×512 contextual crops—transforming the detection dataset into a classification setting. |
| `experiments/evaluation_module/classifier/resnet_classifier.py` | Custom ResNet-18 with training/eval loops, Grad-CAM visualisation, and PR-curve plotting. |
| `experiments/model/train_model.py` | Main entry-point for launching classifier training (baseline or augmented). |
| `experiments/dataset_creation/dataset_assembler.py` | Assembles hybrid datasets by mixing real crops with synthetic images for a target class. |
| `data_generation_backend/diffusion_runner.py` | Versatile Stable Diffusion generation script (txt2img, img2img, ControlNet). |
