from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.fuzzy import membership_high, membership_low, membership_medium
from src.utils import ensure_dir


def plot_memberships(out_dir: Path) -> Path:
    xs = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, [membership_low(x) for x in xs], label="Low")
    ax.plot(xs, [membership_medium(x) for x in xs], label="Medium")
    ax.plot(xs, [membership_high(x) for x in xs], label="High")
    ax.set_xlabel("Normalized indicator")
    ax.set_ylabel("Membership")
    ax.set_title("Fuzzy membership functions")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "membership_functions.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_metric_bars(results: dict, out_dir: Path, metric: str = "f1_macro") -> Path:
    names, values = [], []
    for name, payload in results.items():
        names.append(name)
        values.append(payload["test"][metric])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#4C72B0" if n != "fuzzy_dynamic" else "#C44E52" for n in names]
    ax.bar(names, values, color=colors)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Test-set comparison")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=25)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    path = out_dir / f"comparison_{metric}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_confusion(cm: np.ndarray, class_names: list[str], title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_weight_distribution(weights: np.ndarray, names: list[str], out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([weights[:, i] for i in range(weights.shape[1])])
    ax.set_xticklabels(names)
    ax.set_ylabel("Fusion weight")
    ax.set_title("Fuzzy fusion weights across images")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    path = out_dir / "fusion_weight_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_all_plots(results: dict, class_names: list[str], weights: np.ndarray, extractor_names: list[str], out_dir: str | Path) -> None:
    out_dir = ensure_dir(out_dir)
    plot_memberships(out_dir)
    plot_metric_bars(results, out_dir, "accuracy")
    plot_metric_bars(results, out_dir, "f1_macro")
    plot_weight_distribution(weights, extractor_names, out_dir)
    fuzzy_cm = np.array(results["fuzzy_dynamic"]["test"]["confusion_matrix"])
    plot_confusion(fuzzy_cm, class_names, "Fuzzy dynamic fusion (test)", out_dir / "cm_fuzzy_dynamic.png")
