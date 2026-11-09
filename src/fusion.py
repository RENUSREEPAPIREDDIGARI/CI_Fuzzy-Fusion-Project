from __future__ import annotations

import numpy as np


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, eps, None)


def align_streams(features: dict[str, np.ndarray], names: list[str]) -> dict[str, np.ndarray]:
    """L2-normalize each stream so fusion weights are comparable."""
    return {name: l2_normalize(features[name].astype(np.float64)) for name in names}


def concatenate_streams(aligned: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    return np.concatenate([aligned[name] for name in names], axis=1)


def weighted_concat(aligned: dict[str, np.ndarray], names: list[str], weights: np.ndarray) -> np.ndarray:
    """Scale each feature stream by its (per-sample) fusion weight, then concatenate.

    Concatenation keeps complementary information from every backbone, while
    the fuzzy weights raise or lower that stream's contribution.
    """
    parts = []
    for idx, name in enumerate(names):
        scale = weights[:, idx : idx + 1]
        parts.append(aligned[name] * scale)
    return np.concatenate(parts, axis=1)


def equal_weights(n_samples: int, n_streams: int) -> np.ndarray:
    return np.full((n_samples, n_streams), 1.0 / n_streams, dtype=np.float64)
