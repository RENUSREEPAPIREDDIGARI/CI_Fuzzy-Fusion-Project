"""Create a small synthetic colorectal-style dataset with quality degradations.

Use this only to verify the pipeline. Replace data/raw with your real images
before reporting results.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CLASSES = {
    "adenoma": (180, 70, 90),
    "hyperplastic": (90, 160, 80),
    "normal": (200, 90, 140),
}


def _blob(rng: np.random.Generator, size: int, color: tuple[int, int, int]) -> Image.Image:
    canvas = rng.integers(20, 60, size=(size, size, 3), dtype=np.int32)
    base = np.array(color, dtype=np.int32)
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(8):
        cx, cy = rng.integers(20, size - 20, size=2)
        rx, ry = rng.integers(12, 55, size=2)
        mask = ((xx - cx) ** 2) / (rx**2) + ((yy - cy) ** 2) / (ry**2) <= 1
        shade = base + rng.integers(-30, 30, size=3)
        canvas[mask] = np.clip(shade, 0, 255)
    noise = rng.normal(0, 8, size=canvas.shape)
    rgb = np.clip(canvas + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    for _ in range(4):
        x0, y0 = rng.integers(0, size - 10, size=2)
        x1, y1 = x0 + int(rng.integers(8, 40)), y0 + int(rng.integers(8, 40))
        draw.ellipse([x0, y0, x1, y1], outline=tuple(int(c) for c in np.clip(base - 40, 0, 255)))
    return image


def _degrade(image: Image.Image, rng: random.Random) -> Image.Image:
    mode = rng.choice(["clean", "blur", "dark", "bright", "noise", "low_contrast"])
    if mode == "blur":
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.2, 2.8)))
    elif mode == "dark":
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.35, 0.6))
    elif mode == "bright":
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(1.4, 1.8))
    elif mode == "noise":
        arr = np.asarray(image).astype(np.int16)
        arr = np.clip(arr + rng.randint(8, 22) * np.random.randn(*arr.shape), 0, 255).astype(np.uint8)
        image = Image.fromarray(arr)
    elif mode == "low_contrast":
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.35, 0.6))
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--per-class", type=int, default=40)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    rng_np = np.random.default_rng(args.seed)
    rng_py = random.Random(args.seed)
    for cls, color in CLASSES.items():
        folder = out / cls
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_class):
            image = _blob(rng_np, args.size, color)
            image = _degrade(image, rng_py)
            image.save(folder / f"{cls}_{i:03d}.png")
    print(f"Wrote {args.per_class * len(CLASSES)} dummy images to {out.resolve()}")
    print("Replace this folder with your real colorectal dataset before final experiments.")


if __name__ == "__main__":
    main()
