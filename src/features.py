from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

from src.data import Sample, load_rgb
from src.utils import ensure_dir


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def resolve_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _try_weights(module, weights_enum):
    try:
        return module(weights=weights_enum.IMAGENET1K_V1)
    except Exception:
        try:
            return module(pretrained=True)
        except Exception:
            print(f"Warning: ImageNet weights unavailable for {module.__name__}; using random init.")
            try:
                return module(weights=None)
            except TypeError:
                return module(pretrained=False)


class BackboneExtractor(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        if name == "resnet50":
            net = _try_weights(models.resnet50, models.ResNet50_Weights)
            self.backbone = nn.Sequential(*list(net.children())[:-1])
            self.dim = 2048
        elif name == "densenet121":
            net = _try_weights(models.densenet121, models.DenseNet121_Weights)
            self.backbone = nn.Sequential(net.features, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
            self.dim = 1024
        elif name == "efficientnet_b0":
            net = _try_weights(models.efficientnet_b0, models.EfficientNet_B0_Weights)
            self.backbone = nn.Sequential(net.features, nn.AdaptiveAvgPool2d(1))
            self.dim = 1280
        else:
            raise ValueError(f"Unknown extractor: {name}")
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return torch.flatten(feats, 1)


def build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class FeatureBank:
    def __init__(self, extractor_names: list[str], image_size: int, device: torch.device):
        self.extractor_names = extractor_names
        self.device = device
        self.transform = build_transform(image_size)
        self.extractors = {name: BackboneExtractor(name).to(device) for name in extractor_names}

    @torch.no_grad()
    def extract_one(self, image: Image.Image) -> dict[str, np.ndarray]:
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        out = {}
        for name, model in self.extractors.items():
            vec = model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
            out[name] = vec
        return out

    def extract_dataset(
        self,
        samples: list[Sample],
        cache_dir: str | Path | None = None,
        show_progress: bool = True,
    ) -> dict[str, np.ndarray]:
        cache_path = None
        if cache_dir is not None:
            cache_path = ensure_dir(cache_dir) / "features.npz"
            if cache_path.exists():
                cached = np.load(cache_path, allow_pickle=False)
                expected = {s.path.as_posix() for s in samples}
                if set(cached["paths"]) == expected:
                    print(f"Loaded cached features from {cache_path}")
                    return {name: cached[name] for name in list(self.extractor_names) + ["label_id"]}

        bank = {name: [] for name in self.extractor_names}
        labels = []
        iterator = tqdm(samples, desc="Extracting deep features") if show_progress else samples
        for sample in iterator:
            image = load_rgb(sample.path)
            feats = self.extract_one(image)
            for name, vec in feats.items():
                bank[name].append(vec)
            labels.append(sample.label_id)

        packed = {name: np.stack(vecs, axis=0) for name, vecs in bank.items()}
        packed["label_id"] = np.array(labels, dtype=np.int64)
        if cache_path is not None:
            np.savez_compressed(
                cache_path,
                paths=np.array([s.path.as_posix() for s in samples]),
                **packed,
            )
        return packed
