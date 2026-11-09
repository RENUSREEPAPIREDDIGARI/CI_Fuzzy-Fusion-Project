from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a + 1e-12)
    return (d - x) / (d - c + 1e-12)


def _trimf(x: float, a: float, b: float, c: float) -> float:
    return _trapmf(x, a, b, b, c)


def membership_low(x: float) -> float:
    return _trapmf(x, 0.0, 0.0, 0.20, 0.45)


def membership_medium(x: float) -> float:
    return _trimf(x, 0.25, 0.50, 0.75)


def membership_high(x: float) -> float:
    return _trapmf(x, 0.55, 0.80, 1.0, 1.0)


LINGUISTIC = {
    "low": membership_low,
    "medium": membership_medium,
    "high": membership_high,
}


@dataclass
class FuzzyRule:
    """Sugeno rule: IF antecedents THEN weight vector for (resnet, densenet, efficientnet)."""

    name: str
    antecedents: dict[str, str]  # indicator -> low|medium|high
    weights: np.ndarray
    combine: str = "and"  # and = min, or = max


def default_rules() -> list[FuzzyRule]:
    """Domain rules for colorectal image fusion.

    ResNet-50: robust global semantics under degradation.
    DenseNet-121: fine texture / glandular detail when the image is sharp.
    EfficientNet-B0: balanced multi-scale representation.
    """
    return [
        FuzzyRule(
            "high_quality_texture",
            {"sharpness": "high", "contrast": "high", "entropy": "high"},
            np.array([0.20, 0.45, 0.35]),
        ),
        FuzzyRule(
            "sharp_and_clean",
            {"sharpness": "high", "cleanliness": "high"},
            np.array([0.25, 0.40, 0.35]),
        ),
        FuzzyRule(
            "blurry",
            {"sharpness": "low"},
            np.array([0.55, 0.20, 0.25]),
        ),
        FuzzyRule(
            "noisy",
            {"cleanliness": "low"},
            np.array([0.50, 0.20, 0.30]),
        ),
        FuzzyRule(
            "poor_illumination",
            {"illumination": "low"},
            np.array([0.48, 0.22, 0.30]),
        ),
        FuzzyRule(
            "low_contrast",
            {"contrast": "low"},
            np.array([0.42, 0.23, 0.35]),
        ),
        FuzzyRule(
            "low_entropy",
            {"entropy": "low"},
            np.array([0.45, 0.20, 0.35]),
        ),
        FuzzyRule(
            "medium_quality",
            {"sharpness": "medium", "contrast": "medium"},
            np.array([0.34, 0.33, 0.33]),
        ),
        FuzzyRule(
            "high_entropy_medium_sharp",
            {"entropy": "high", "sharpness": "medium"},
            np.array([0.28, 0.38, 0.34]),
        ),
    ]


class FuzzyFusionEngine:
    def __init__(self, extractor_names: list[str], rules: list[FuzzyRule] | None = None):
        self.extractor_names = extractor_names
        self.rules = rules or default_rules()
        self.n_streams = len(extractor_names)

    def fuzzify(self, indicators: dict[str, float]) -> dict[str, dict[str, float]]:
        return {
            name: {label: fn(value) for label, fn in LINGUISTIC.items()}
            for name, value in indicators.items()
        }

    def firing_strength(self, rule: FuzzyRule, fuzzy_set: dict[str, dict[str, float]]) -> float:
        degrees = []
        for indicator, label in rule.antecedents.items():
            degrees.append(fuzzy_set.get(indicator, {}).get(label, 0.0))
        if not degrees:
            return 0.0
        if rule.combine == "or":
            return float(max(degrees))
        return float(min(degrees))

    def infer(self, indicators: dict[str, float]) -> np.ndarray:
        fuzzy_set = self.fuzzify(indicators)
        numer = np.zeros(self.n_streams, dtype=np.float64)
        denom = 0.0
        for rule in self.rules:
            strength = self.firing_strength(rule, fuzzy_set)
            if strength <= 0:
                continue
            w = np.asarray(rule.weights, dtype=np.float64)
            if w.size != self.n_streams:
                raise ValueError("Rule weight length must match number of extractors")
            numer += strength * w
            denom += strength
        if denom <= 1e-12:
            weights = np.ones(self.n_streams, dtype=np.float64) / self.n_streams
        else:
            weights = numer / denom
        weights = np.clip(weights, 1e-6, None)
        weights = weights / weights.sum()
        return weights

    def infer_batch(self, indicator_list: list[dict[str, float]]) -> np.ndarray:
        return np.stack([self.infer(item) for item in indicator_list], axis=0)
