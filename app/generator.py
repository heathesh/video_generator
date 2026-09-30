import gc
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageOps

from app.config import Settings

# Let PyTorch fall back to the CPU for any op not yet implemented on Apple's MPS backend.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

logger = logging.getLogger(__name__)

# LTX-Video requires spatial dims divisible by 32 and a frame count of the form 8k + 1.
SPATIAL_MULTIPLE = 32
FRAME_MULTIPLE = 8
MIN_NUM_FRAMES = FRAME_MULTIPLE + 1


@dataclass(frozen=True)
class VideoParams:
    prompt: str
    negative_prompt: str
    width: int
    height: int
    num_frames: int
    num_inference_steps: int
    guidance_scale: float
    seed: int
    fps: int


def snap_dimension(value: int) -> int:
    """Round down to the nearest multiple of 32 (minimum 32)."""
    return max(SPATIAL_MULTIPLE, value // SPATIAL_MULTIPLE * SPATIAL_MULTIPLE)


def snap_num_frames(value: int) -> int:
    """Round down to the nearest 8k + 1 frame count (minimum 9)."""
    return max(MIN_NUM_FRAMES, (value - 1) // FRAME_MULTIPLE * FRAME_MULTIPLE + 1)


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    """Resize and center-crop an image so it exactly fills width x height."""
    return ImageOps.fit(image.convert("RGB"), (width, height), Image.Resampling.LANCZOS)


class Generator(Protocol):
    """Anything that can turn VideoParams (+ optional first-frame image) into an mp4 on disk."""

    model_id: str
    device: str
    loaded: bool

    def load(self) -> None: ...

    def generate(self, params: VideoParams, image: Image.Image | None, output_path: Path) -> Path: ...


class LTXVideoGenerator:
    """Wraps the diffusers LTXConditionPipeline for text-to-video and image-to-video on Apple Silicon."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_id = settings.model_id
        self.device = self._resolve_device(settings.device)
        self.loaded = False
        self._pipe = None

    @staticmethod
    def _resolve_device(requested: str) -> str:
        import torch

        if requested != "auto":
            return requested
        return "mps" if torch.backends.mps.is_available() else "cpu"

    def load(self) -> None:
        import torch
        from diffusers import LTXConditionPipeline

        dtype = getattr(torch, self.settings.dtype)
        logger.info("Loading %s (%s) on %s ...", self.model_id, self.settings.dtype, self.device)
        pipe = LTXConditionPipeline.from_pretrained(self.model_id, dtype=dtype)
        pipe.to(self.device)
        # Decode the video latents in tiles to keep peak memory down.
        pipe.vae.enable_tiling()
        self._pipe = pipe
        self.loaded = True
        logger.info("Model loaded.")

    def generate(self, params: VideoParams, image: Image.Image | None, output_path: Path) -> Path:
        import torch
        from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
        from diffusers.utils import export_to_video

        if self._pipe is None:
            raise RuntimeError("Model is not loaded")

        conditions = None
        if image is not None:
            first_frame = fit_image(image, params.width, params.height)
            conditions = [LTXVideoCondition(image=first_frame, frame_index=0)]

        # A CPU generator keeps seeds reproducible regardless of device.
        generator = torch.Generator(device="cpu").manual_seed(params.seed)
        try:
            frames = self._pipe(
                conditions=conditions,
                prompt=params.prompt,
                negative_prompt=params.negative_prompt,
                width=params.width,
                height=params.height,
                num_frames=params.num_frames,
                frame_rate=params.fps,
                num_inference_steps=params.num_inference_steps,
                guidance_scale=params.guidance_scale,
                decode_timestep=0.05,
                decode_noise_scale=0.025,
                image_cond_noise_scale=0.025,
                generator=generator,
            ).frames[0]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            export_to_video(frames, str(output_path), fps=params.fps)
        finally:
            gc.collect()
            if self.device == "mps":
                torch.mps.empty_cache()
        return output_path
