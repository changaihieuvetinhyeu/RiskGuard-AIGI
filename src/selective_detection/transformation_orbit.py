"""Deterministic Phase 4 five-view transformation orbit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from PIL import Image, ImageFilter


ORBIT_VERSION = "phase4_orbit_v1"
ORBIT_SEED = 20260916
VIEW_ORDER = (
    "identity",
    "jpeg_q75",
    "resize_075_restore",
    "gaussian_blur_sigma_05",
    "center_crop_090_restore",
)


@dataclass(frozen=True)
class OrbitView:
    view_name: str
    parameters: dict[str, Any]

    @property
    def parameter_json(self) -> str:
        return canonical_json(self.parameters)


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def default_orbit_views() -> tuple[OrbitView, ...]:
    return (
        OrbitView("identity", {"operation": "identity"}),
        OrbitView("jpeg_q75", {"operation": "jpeg", "quality": 75, "subsampling": "4:2:0"}),
        OrbitView(
            "resize_075_restore",
            {"operation": "resize_restore", "scale": 0.75, "resample": "bicubic", "antialias": True},
        ),
        OrbitView("gaussian_blur_sigma_05", {"operation": "gaussian_blur", "sigma": 0.5}),
        OrbitView(
            "center_crop_090_restore",
            {"operation": "center_crop_restore", "crop_fraction": 0.9, "resample": "bicubic", "antialias": True},
        ),
    )


def orbit_config_payload() -> dict[str, Any]:
    return {
        "orbit_version": ORBIT_VERSION,
        "seed": ORBIT_SEED,
        "views": [{"view_name": view.view_name, "parameters": view.parameters} for view in default_orbit_views()],
    }


def transform_chain_id(view: OrbitView, orbit_version: str = ORBIT_VERSION) -> str:
    return hashlib.sha256(f"{orbit_version}{view.view_name}{view.parameter_json}".encode("utf-8")).hexdigest()


def make_view_id(parent_sample_id: str, parent_sha256: str, chain_id: str) -> str:
    return hashlib.sha256(f"{parent_sample_id}{parent_sha256}{chain_id}".encode("utf-8")).hexdigest()


def _rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def identity(image: Image.Image) -> Image.Image:
    return _rgb(image).copy()


def jpeg_quality(image: Image.Image, quality: int = 75) -> Image.Image:
    buffer = BytesIO()
    _rgb(image).save(buffer, format="JPEG", quality=quality, subsampling=2)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def downscale_restore(image: Image.Image, scale: float = 0.75) -> Image.Image:
    image = _rgb(image)
    width, height = image.size
    small = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.BICUBIC,
        reducing_gap=3.0,
    )
    return small.resize((width, height), Image.Resampling.BICUBIC, reducing_gap=3.0)


def gaussian_blur(image: Image.Image, sigma: float = 0.5) -> Image.Image:
    return _rgb(image).filter(ImageFilter.GaussianBlur(radius=sigma))


def center_crop_resize(image: Image.Image, crop_fraction: float = 0.90) -> Image.Image:
    image = _rgb(image)
    width, height = image.size
    crop_w = max(1, round(width * crop_fraction))
    crop_h = max(1, round(height * crop_fraction))
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    crop = image.crop((left, top, left + crop_w, top + crop_h))
    return crop.resize((width, height), Image.Resampling.BICUBIC, reducing_gap=3.0)


def apply_view(image: Image.Image, view_name: str) -> Image.Image:
    if view_name == "identity":
        return identity(image)
    if view_name == "jpeg_q75":
        return jpeg_quality(image, quality=75)
    if view_name == "resize_075_restore":
        return downscale_restore(image, scale=0.75)
    if view_name == "gaussian_blur_sigma_05":
        return gaussian_blur(image, sigma=0.5)
    if view_name == "center_crop_090_restore":
        return center_crop_resize(image, crop_fraction=0.90)
    raise ValueError(f"unknown Phase 4 orbit view: {view_name}")


def transformed_pixel_sha256(image: Image.Image) -> str:
    rgb = _rgb(image)
    digest = hashlib.sha256()
    digest.update(rgb.mode.encode("ascii"))
    digest.update(str(rgb.size[0]).encode("ascii"))
    digest.update(b"x")
    digest.update(str(rgb.size[1]).encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def build_default_orbit(image: Image.Image) -> list[Image.Image]:
    """Return the SOICT five-view probe orbit in the frozen order."""
    return [apply_view(image, view_name) for view_name in VIEW_ORDER]
