from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class QualityVector:
    sharpness: float
    contrast: float
    entropy: float
    noise: float
    brightness: float

    def as_dict(self) -> dict[str, float]:
        return {
            "sharpness": self.sharpness,
            "contrast": self.contrast,
            "entropy": self.entropy,
            "noise": self.noise,
            "brightness": self.brightness,
        }

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.sharpness, self.contrast, self.entropy, self.noise, self.brightness],
            dtype=np.float64,
        )


def _to_gray(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return gray


def sharpness_score(gray: np.ndarray) -> float:
    """Variance of Laplacian — higher means sharper."""
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)
    return float(lap.var())


def contrast_score(gray: np.ndarray) -> float:
    """RMS contrast."""
    mean = float(gray.mean())
    return float(np.sqrt(np.mean((gray - mean) ** 2)))


def entropy_score(gray: np.ndarray) -> float:
    """Shannon entropy of the 8-bit histogram."""
    hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [256], [0, 256]).ravel()
    hist = hist / (hist.sum() + 1e-12)
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum())


def noise_score(gray: np.ndarray) -> float:
    """High-frequency residual after Gaussian smoothing. Higher = noisier."""
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 1.0)
    residual = gray.astype(np.float32) - blurred
    return float(residual.std())


def brightness_score(gray: np.ndarray) -> float:
    """Mean intensity in [0, 255]."""
    return float(gray.mean())


def compute_quality(image: Image.Image) -> QualityVector:
    gray = _to_gray(image)
    return QualityVector(
        sharpness=sharpness_score(gray),
        contrast=contrast_score(gray),
        entropy=entropy_score(gray),
        noise=noise_score(gray),
        brightness=brightness_score(gray),
    )


class QualityNormalizer:
    """Map raw IQA scores to [0, 1] using dataset percentiles.

    Brightness is converted to *illumination quality*: values near the
    dataset median become 1, extremes (too dark / too bright) become 0.
    Noise is inverted so that 1 means clean.
    """

    def __init__(self, clip_low: float = 2.0, clip_high: float = 98.0):
        self.clip_low = clip_low
        self.clip_high = clip_high
        self.stats: dict[str, tuple[float, float]] = {}
        self.brightness_median = 127.0

    def fit(self, qualities: list[QualityVector]) -> "QualityNormalizer":
        keys = ["sharpness", "contrast", "entropy", "noise", "brightness"]
        matrix = np.stack([q.as_array() for q in qualities], axis=0)
        for idx, key in enumerate(keys):
            lo = float(np.percentile(matrix[:, idx], self.clip_low))
            hi = float(np.percentile(matrix[:, idx], self.clip_high))
            if hi <= lo:
                hi = lo + 1e-6
            self.stats[key] = (lo, hi)
        self.brightness_median = float(np.median(matrix[:, 4]))
        return self

    def _minmax(self, value: float, key: str) -> float:
        lo, hi = self.stats[key]
        return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))

    def transform_one(self, quality: QualityVector) -> dict[str, float]:
        sharpness = self._minmax(quality.sharpness, "sharpness")
        contrast = self._minmax(quality.contrast, "contrast")
        entropy = self._minmax(quality.entropy, "entropy")
        noise_raw = self._minmax(quality.noise, "noise")
        brightness_raw = quality.brightness
        lo, hi = self.stats["brightness"]
        span = max(hi - lo, 1e-6)
        deviation = abs(brightness_raw - self.brightness_median) / (0.5 * span)
        illumination = float(np.clip(1.0 - deviation, 0.0, 1.0))
        return {
            "sharpness": sharpness,
            "contrast": contrast,
            "entropy": entropy,
            "cleanliness": float(1.0 - noise_raw),  # 1 = low noise
            "illumination": illumination,
        }

    def transform(self, qualities: list[QualityVector]) -> list[dict[str, float]]:
        return [self.transform_one(q) for q in qualities]
