from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def make_split(labels: np.ndarray, train_ratio: float, val_ratio: float, seed: int) -> Split:
    indices = np.arange(len(labels))
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices,
        labels,
        test_size=1.0 - train_ratio,
        random_state=seed,
        stratify=labels if _can_stratify(labels) else None,
    )
    relative_val = val_ratio / (val_ratio + (1.0 - train_ratio - val_ratio))
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1.0 - relative_val,
        random_state=seed,
        stratify=y_temp if _can_stratify(y_temp) else None,
    )
    return Split(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def _can_stratify(labels: np.ndarray) -> bool:
    _, counts = np.unique(labels, return_counts=True)
    return counts.min() >= 2 and len(counts) > 1


def build_classifier(kind: str, C: float, max_iter: int, seed: int):
    if kind == "linear_svm":
        return LinearSVC(C=C, max_iter=max_iter, dual=False, random_state=seed)
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=seed,
        solver="lbfgs",
    )


def train_and_score(
    X: np.ndarray,
    y: np.ndarray,
    split: Split,
    class_names: list[str],
    kind: str,
    C: float,
    max_iter: int,
    seed: int,
) -> dict:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[split.train_idx])
    X_val = scaler.transform(X[split.val_idx])
    X_test = scaler.transform(X[split.test_idx])
    y_train, y_val, y_test = y[split.train_idx], y[split.val_idx], y[split.test_idx]

    model = build_classifier(kind, C, max_iter, seed)
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    def pack(y_true, y_pred, split_name: str) -> dict:
        return {
            "split": split_name,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "report": classification_report(
                y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
            ),
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
        }

    return {"val": pack(y_val, val_pred, "val"), "test": pack(y_test, test_pred, "test")}
