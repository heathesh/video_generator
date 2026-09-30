from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Request limits
    max_width: int = 1280
    max_height: int = 1280
    max_num_frames: int = 257
    max_num_inference_steps: int = 100
    max_upload_mb: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
