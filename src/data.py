from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Sample:
    path: Path
    label: str
    label_id: int


def discover_dataset(root: str | Path) -> tuple[list[Sample], list[str]]:
    """Load a class-folder dataset: data/raw/<class_name>/*.jpg

    Also accepts a single extra nesting level (e.g. train/class/*.jpg is
    flattened by using the immediate parent folder as the class name).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {root}\n"
            "Place images in class subfolders, for example:\n"
            "  data/raw/adenoma/*.jpg\n"
            "  data/raw/hyperplastic/*.jpg\n"
            "  data/raw/normal/*.jpg\n"
            "Or run: python scripts/generate_dummy_data.py"
        )

    files: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(root.rglob(f"*{ext}"))
        files.extend(root.rglob(f"*{ext.upper()}"))
    files = sorted({p.resolve() for p in files if p.is_file()})
    if not files:
        raise FileNotFoundError(
            f"No images found under {root}. Supported types: {sorted(IMAGE_EXTENSIONS)}"
        )

    labels = []
    for path in files:
        if path.parent.resolve() == root.resolve():
            raise ValueError(
                "Images must sit in class subfolders, not directly in the dataset root.\n"
                f"Found: {path}"
            )
        labels.append(path.parent.name)

    class_names = sorted(set(labels))
    name_to_id = {name: idx for idx, name in enumerate(class_names)}
    samples = [
        Sample(path=path, label=label, label_id=name_to_id[label])
        for path, label in zip(files, labels)
    ]
    return samples, class_names


def load_rgb(path: str | Path, size: int | None = None) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if size is not None:
        image = image.resize((size, size), Image.Resampling.BILINEAR)
    return image
