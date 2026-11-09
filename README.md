# Fuzzy-Guided Dynamic Feature Fusion

Colorectal images do not all carry equally reliable visual information. Blur, poor lighting, low contrast, and noise change how trustworthy each feature extractor is. This project estimates per-image quality, maps those scores through a fuzzy rule base, and uses the resulting weights to fuse three frozen ImageNet backbones. Dynamic fusion is compared with equal-weight fusion and with each backbone used alone.

## What was implemented

1. **Deep features** from ResNet-50, DenseNet-121, and EfficientNet-B0 (global average pooled, L2-normalized).
2. **No-reference quality indicators**: sharpness (Laplacian variance), RMS contrast, histogram entropy, residual noise, and illumination quality.
3. **Fuzzy membership functions** Low / Medium / High on each normalized indicator.
4. **Sugeno fuzzy rules** that output a 3-stream weight vector (robust semantics vs texture vs multi-scale).
5. **Baselines**: each single extractor, fixed equal weights, and fuzzy dynamic weights.
6. **Linear classifier** (logistic regression) trained on the fused vectors; accuracy / precision / recall / F1 and confusion matrices are saved.

## Dataset layout (required)

Put your colorectal images in class folders:

```text
data/raw/
  adenoma/*.jpg
  hyperplastic/*.jpg
  serrated/*.jpg
  normal/*.jpg
```

Class names can be anything; the folder name becomes the label. Nested folders still work as long as the **parent folder of each image** is the class name (for example `data/raw/train/adenoma/img01.jpg` is labeled `adenoma`).

Supported types: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`.

## How to run

```bash
python -m pip install -r requirements.txt
python scripts/run_experiment.py
```

## Interactive dashboard

Install the requirements, then launch the local web interface:

```bash
python -m streamlit run scripts/dashboard.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`). In the
sidebar, choose the dataset folder on your computer. The dashboard checks the
class-folder layout, runs the experiment, displays progress, and shows the
comparison metrics, plots, and per-image quality/fusion weights. The dataset
is processed locally and is not uploaded.

Optional dummy data (pipeline check only, not for reporting):

```bash
python scripts/generate_dummy_data.py
python scripts/run_experiment.py
```

Override the data folder if you stored images somewhere else:

```bash
python scripts/run_experiment.py --data "path/to/your/dataset"
```

First run downloads ImageNet weights and extracts features (cached under `results/feature_cache`). Later runs reuse the cache if the image set is unchanged.

## Outputs

| File | Contents |
|------|----------|
| `results/metrics_summary.csv` | Comparison table |
| `results/metrics.json` | Full numeric report |
| `results/per_image_quality_and_weights.csv` | Indicators and fuzzy weights per image |
| `results/plots/` | Membership curves, metric bars, weight boxplots, confusion matrix |

## Fuzzy idea (short)

- Sharp, high-contrast, high-entropy images raise DenseNet / EfficientNet weights (texture and multi-scale detail).
- Blur, noise, or poor illumination raise ResNet-50 weight (more stable global semantics).
- Medium quality stays close to equal fusion.

Weights are applied **per image** before concatenation, then a single classifier is trained. The same train/val/test split is used for every method so the comparison is fair.

## Notes for your report

- Report results on the **real** dataset, not dummy images.
- If a class has very few images, stratified split may be unstable; add more samples or merge rare classes.
- GPU is used automatically when CUDA is available; CPU works but feature extraction is slower.
