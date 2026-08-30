"""Detector adapter metadata and smoke-readiness checks."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DetectorCheck:
    name: str
    status: str
    missing_files: tuple[str, ...]
    missing_packages: tuple[str, ...]
    command: tuple[str, ...]
    message: str

    @property
    def ready(self) -> bool:
        return self.status == "ready"


@dataclass(frozen=True)
class DetectorAdapter:
    name: str
    repo_path: Path
    checkpoint_path: Path
    entrypoint: Path
    required_packages: tuple[str, ...]
    smoke_command: tuple[str, ...]

    def check(self) -> DetectorCheck:
        missing_files = tuple(
            str(path)
            for path in (self.repo_path, self.checkpoint_path, self.entrypoint)
            if not path.exists()
        )
        missing_packages = tuple(
            package for package in self.required_packages if importlib.util.find_spec(package) is None
        )
        if missing_files:
            status = "blocked_missing_files"
            message = "Official repository, checkpoint, or entrypoint is missing."
        elif missing_packages:
            status = "blocked_missing_dependencies"
            message = "Install the detector runtime dependencies in an isolated environment before inference."
        else:
            status = "ready"
            message = "Official files and Python dependencies are available for smoke inference."
        return DetectorCheck(
            name=self.name,
            status=status,
            missing_files=missing_files,
            missing_packages=missing_packages,
            command=self.smoke_command,
            message=message,
        )


def default_adapters(project_root: str | Path = ".") -> tuple[DetectorAdapter, ...]:
    root = Path(project_root)
    return (
        DetectorAdapter(
            name="univfd",
            repo_path=root / "third_party/UniversalFakeDetect",
            checkpoint_path=root / "third_party/UniversalFakeDetect/pretrained_weights/fc_weights.pth",
            entrypoint=root / "third_party/UniversalFakeDetect/validate.py",
            required_packages=("torch", "torchvision", "numpy", "PIL", "sklearn", "scipy"),
            smoke_command=(
                "python",
                "third_party/UniversalFakeDetect/validate.py",
                "--arch=CLIP:ViT-L/14",
                "--ckpt=third_party/UniversalFakeDetect/pretrained_weights/fc_weights.pth",
                "--real_path=<pilot-real-dir>",
                "--fake_path=<pilot-fake-dir>",
                "--data_mode=ours",
                "--max_sample=8",
            ),
        ),
        DetectorAdapter(
            name="safe",
            repo_path=root / "third_party/SAFE",
            checkpoint_path=root / "third_party/SAFE/checkpoint/checkpoint-best.pth",
            entrypoint=root / "third_party/SAFE/main_finetune.py",
            required_packages=("torch", "torchvision", "numpy", "PIL", "timm"),
            smoke_command=(
                "python",
                "third_party/SAFE/main_finetune.py",
                "--eval",
                "true",
                "--model",
                "SAFE",
                "--resume",
                "third_party/SAFE/checkpoint/checkpoint-best.pth",
                "--eval_data_path=<pilot-image-folder>",
                "--device",
                "cpu",
            ),
        ),
    )
