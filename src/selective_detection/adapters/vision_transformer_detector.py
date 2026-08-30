"""Adapter for the official UniversalFakeDetect/UnivFD checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image
from torchvision import transforms


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class UnivFDDetector:
    name = "univfd"
    preprocessing_id = "univfd_clip_vitl14_centercrop224_clip_normalize"
    official_probability_threshold = 0.5
    official_raw_logit_threshold = 0.0

    def __init__(
        self,
        repo_path: str | Path = "third_party/UniversalFakeDetect",
        checkpoint_path: str | Path = "third_party/UniversalFakeDetect/pretrained_weights/fc_weights.pth",
        device: str | torch.device | None = None,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        sys.path.insert(0, str(self.repo_path))
        from models import get_model  # type: ignore[import-not-found]

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = get_model("CLIP:ViT-L/14")
        state = torch.load(self.checkpoint_path, map_location="cpu")
        self.model.fc.load_state_dict(state)
        self.model.eval().to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ]
        )
        self.embedding_dimension = 768

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

    @torch.inference_mode()
    def predict(self, images: Sequence[str | Path | Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self._load_batch(images)
        embeddings = self.model.model.encode_image(batch).float()
        logits = self.model.fc(embeddings).flatten().float()
        return logits.detach().cpu(), embeddings.detach().cpu()

    def predict_table(
        self,
        sample_ids: Sequence[str],
        image_paths: Sequence[str | Path],
        checkpoint_sha256: str,
        batch_size: int = 16,
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
