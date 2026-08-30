"""Adapter for the official SAFE checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torchvision import transforms


class SAFEDetector:
    name = "safe"
    preprocessing_id = "safe_resnet50_dwt_centercrop256_totensor"
    official_probability_threshold = 0.5
    official_raw_logit_threshold = 0.0

    def __init__(
        self,
        repo_path: str | Path = "third_party/SAFE",
        checkpoint_path: str | Path = "third_party/SAFE/checkpoint/checkpoint-best.pth",
        device: str | torch.device | None = None,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        sys.path.insert(0, str(self.repo_path))
        from models.resnet import resnet50  # type: ignore[import-not-found]

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = resnet50(num_classes=2)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval().to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop([256, 256]),
                transforms.ToTensor(),
            ]
        )
        self.embedding_dimension = 512

    def _load_batch(self, images: Sequence[str | Path | Image.Image]) -> torch.Tensor:
        tensors = []
        for image in images:
            if isinstance(image, Image.Image):
                pil_image = image.convert("RGB")
            else:
                with Image.open(image) as handle:
                    pil_image = handle.convert("RGB")
            tensors.append(self.transform(pil_image))
        return torch.stack(tensors, dim=0).to(self.device)

    def _features_and_logits(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = 1 * self.model._preprocess_dwt(batch)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.avgpool(x)
        embeddings = x.view(x.size(0), -1)
        two_class_logits = self.model.fc1(embeddings)
        raw_logits = (two_class_logits[:, 1] - two_class_logits[:, 0]).float()
        return raw_logits, embeddings.float()

    @torch.inference_mode()
    def predict(self, images: Sequence[str | Path | Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self._load_batch(images)
        logits, embeddings = self._features_and_logits(batch)
        return logits.detach().cpu(), embeddings.detach().cpu()

    def predict_table(
        self,
        sample_ids: Sequence[str],
        image_paths: Sequence[str | Path],
        checkpoint_sha256: str,
        batch_size: int = 32,
    ):
        import pandas as pd

        rows = []
        embedding_batches = []
        for start in range(0, len(image_paths), batch_size):
            end = min(start + batch_size, len(image_paths))
            logits, embeddings = self.predict(image_paths[start:end])
            probabilities = torch.sigmoid(logits)
            embedding_batches.append(embeddings)
            for offset, sample_id in enumerate(sample_ids[start:end]):
                logit = float(logits[offset].item())
                probability = float(probabilities[offset].item())
                rows.append(
                    {
                        "sample_id": sample_id,
                        "raw_logit": logit,
                        "fake_probability": probability,
                        "predicted_label": int(probability >= self.official_probability_threshold),
                        "embedding_dimension": self.embedding_dimension,
                        "checkpoint_sha256": checkpoint_sha256,
                        "preprocessing_id": self.preprocessing_id,
                    }
                )
        return pd.DataFrame(rows), torch.cat(embedding_batches, dim=0).numpy()
