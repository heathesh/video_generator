from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True)
class AudioBackendInfo:
    max_seconds: float  # longest clip the backend can score
    num_inference_steps: int
    guidance_scale: float


# Per-backend limits and defaults. Kept here, not in app/audio.py, so the API can validate requests without importing
# any backend.
AUDIO_BACKENDS: dict[str, AudioBackendInfo] = {
    "stable-audio": AudioBackendInfo(max_seconds=47.0, num_inference_steps=100, guidance_scale=7.0),
    # Trained on 8 s clips; quality drops well beyond that.
    "mmaudio": AudioBackendInfo(max_seconds=10.0, num_inference_steps=25, guidance_scale=4.5),
    # Its feature extractor reads at most 15 s of video.
    "hunyuan-foley": AudioBackendInfo(max_seconds=15.0, num_inference_steps=50, guidance_scale=4.5),
}

AudioBackendName = Literal["none", "stable-audio", "mmaudio", "hunyuan-foley"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration. Every field can be overridden with a `VIDEO_<FIELD>` env var or a `.env` file."""

    model_config = SettingsConfigDict(
        env_prefix="VIDEO_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    # Model
    model_id: str = "Lightricks/LTX-Video-0.9.5"
    dtype: str = "bfloat16"  # bfloat16 | float16 | float32
    device: str = "auto"  # auto | mps | cpu

    # Audio (chosen with `make setup-audio`; loaded on the first job that asks for audio)
    audio_backend: AudioBackendName = "none"
    audio_dtype: str = "bfloat16"  # bfloat16 | float32
    stable_audio_model_id: str = "stabilityai/stable-audio-open-1.0"
    audio_variant: str = "large_44k_v2"  # MMAudio: small_16k | small_44k | medium_44k | large_44k | large_44k_v2
    audio_weights_dir: Path = Path.home() / ".cache" / "mmaudio"  # MMAudio weights
    hunyuan_foley_dir: Path = PROJECT_ROOT / "backends" / "hunyuan_foley"  # its isolated env, runner and source
    hunyuan_foley_weights_dir: Path = Path.home() / ".cache" / "hunyuan-foley"
    hunyuan_foley_offload: bool = True  # load its sub-models on demand to lower peak memory (~14 GB measured with it on)

    # Storage
    output_dir: Path = Path("outputs")

    # Generation defaults (used when a request omits a field)
    default_width: int = 704
    default_height: int = 480
    default_num_frames: int = 97  # ~4 seconds at 24 fps
    default_num_inference_steps: int = 40
    default_guidance_scale: float = 3.0
    default_fps: int = 24
    default_negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted"
    default_audio_num_inference_steps: int | None = None  # None: the backend's default
    default_audio_guidance_scale: float | None = None  # None: the backend's default
    default_audio_negative_prompt: str = ""

    # Request limits
    max_width: int = 1280
    max_height: int = 1280
    max_num_frames: int = 257
    max_num_inference_steps: int = 100
    max_upload_mb: int = 20


    @property
    def audio(self) -> AudioBackendInfo | None:
        """Limits and defaults of the configured audio backend (with overrides applied), or None if audio is off."""
        info = AUDIO_BACKENDS.get(self.audio_backend)
        if info is None:
            return None
        return AudioBackendInfo(
            max_seconds=info.max_seconds,
            num_inference_steps=self.default_audio_num_inference_steps or info.num_inference_steps,
            guidance_scale=self.default_audio_guidance_scale or info.guidance_scale,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
