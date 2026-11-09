from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data import discover_dataset, load_rgb
from src.evaluate import make_split, train_and_score
from src.features import FeatureBank, resolve_device
from src.fusion import align_streams, equal_weights, weighted_concat
from src.fuzzy import FuzzyFusionEngine
from src.quality import QualityNormalizer, compute_quality
from src.utils import ensure_dir, load_config, set_seed
from src.visualize import save_all_plots


def collect_quality(samples, image_size: int):
    raw = []
    for sample in tqdm(samples, desc="Computing image-quality indicators"):
        image = load_rgb(sample.path, size=image_size)
        raw.append(compute_quality(image))
    return raw


def run_experiment(config_path: str | Path = "config.yaml", data_root: str | Path | None = None) -> dict:
    config = load_config(config_path)
    set_seed(config["seed"])
    out_dir = ensure_dir(config["output"]["dir"])
    plots_dir = ensure_dir(out_dir / "plots")

    root = Path(data_root) if data_root else Path(config["data"]["root"])
    samples, class_names = discover_dataset(root)
    print(f"Found {len(samples)} images in {len(class_names)} classes: {class_names}")

    extractor_names = config["features"]["extractors"]
    device = resolve_device(config["features"]["device"])
    print(f"Using device: {device}")

    raw_quality = collect_quality(samples, config["data"]["image_size"])
    normalizer = QualityNormalizer(
        clip_low=config["quality"]["clip_low"],
        clip_high=config["quality"]["clip_high"],
    ).fit(raw_quality)
    indicators = normalizer.transform(raw_quality)

    engine = FuzzyFusionEngine(extractor_names)
    fuzzy_weights = engine.infer_batch(indicators)

    bank = FeatureBank(extractor_names, config["data"]["image_size"], device)
    packed = bank.extract_dataset(samples, cache_dir=config["features"]["cache_dir"])
    y = packed["label_id"]
    aligned = align_streams(packed, extractor_names)

    methods: dict[str, np.ndarray] = {}
    for name in extractor_names:
        methods[f"single_{name}"] = aligned[name]

    eq = equal_weights(len(samples), len(extractor_names))
    methods["fixed_equal"] = weighted_concat(aligned, extractor_names, eq)
    methods["fuzzy_dynamic"] = weighted_concat(aligned, extractor_names, fuzzy_weights)

    split = make_split(
        y,
        config["data"]["train_ratio"],
        config["data"]["val_ratio"],
        config["seed"],
    )

    results = {}
    for method_name, X in methods.items():
        print(f"Training classifier for: {method_name}")
        results[method_name] = train_and_score(
            X,
            y,
            split,
            class_names,
            kind=config["classifier"]["type"],
            C=config["classifier"]["C"],
            max_iter=config["classifier"]["max_iter"],
            seed=config["seed"],
        )

    quality_rows = []
    for sample, raw, ind, w in zip(samples, raw_quality, indicators, fuzzy_weights):
        row = {
            "path": sample.path.as_posix(),
            "label": sample.label,
            **{f"raw_{k}": v for k, v in raw.as_dict().items()},
            **{f"norm_{k}": v for k, v in ind.items()},
        }
        for i, name in enumerate(extractor_names):
            row[f"w_{name}"] = float(w[i])
        quality_rows.append(row)
    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(out_dir / "per_image_quality_and_weights.csv", index=False)

    summary_rows = []
    for name, payload in results.items():
        summary_rows.append(
            {
                "method": name,
                "test_accuracy": payload["test"]["accuracy"],
                "test_precision_macro": payload["test"]["precision_macro"],
                "test_recall_macro": payload["test"]["recall_macro"],
                "test_f1_macro": payload["test"]["f1_macro"],
                "val_accuracy": payload["val"]["accuracy"],
                "val_f1_macro": payload["val"]["f1_macro"],
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("test_f1_macro", ascending=False)
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)

    serializable = {
        "class_names": class_names,
        "n_images": len(samples),
        "extractors": extractor_names,
        "metrics": summary.to_dict(orient="records"),
        "details": {
            name: {"val": {k: v for k, v in p["val"].items() if k not in {"y_true", "y_pred", "report"}},
                   "test": {k: v for k, v in p["test"].items() if k not in {"y_true", "y_pred", "report"}}}
            for name, p in results.items()
        },
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)

    save_all_plots(results, class_names, fuzzy_weights, extractor_names, plots_dir)
    print("\n=== Test metrics ===")
    print(summary.to_string(index=False))
    print(f"\nSaved results to {out_dir.resolve()}")
    return {"results": results, "summary": summary, "class_names": class_names}
