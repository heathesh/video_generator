"""Pre-download the model weights so the first API start doesn't block on a multi-GB download.

Usage:
  uv run python scripts/download_model.py [MODEL_ID]              # video model, into ~/.cache/huggingface
  uv run python scripts/download_model.py --audio=stable-audio    # Stable Audio Open, into ~/.cache/huggingface
  uv run python scripts/download_model.py --audio=mmaudio         # MMAudio, into VIDEO_AUDIO_WEIGHTS_DIR (~/.cache/mmaudio)
  uv run python scripts/download_model.py --audio=hunyuan-foley   # into VIDEO_HUNYUAN_FOLEY_WEIGHTS_DIR + encoders

`make setup-audio` runs the --audio form for you.
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402


def download_hf_model(model_id: str) -> None:
    print(f"Downloading {model_id} (this can take a while; progress bars below)...")
    path = snapshot_download(
        model_id,
        # Weights, configs and tokenizer only; skip example media and single-file checkpoints.
        allow_patterns=["*.json", "*.txt", "*.model", "*/*.safetensors"],
    )
    size_gb = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e9
    print(f"Done: {path} ({size_gb:.1f} GB)")


def download_mmaudio() -> None:
    from app.audio import mmaudio_model_config

    settings = get_settings()
    cfg = mmaudio_model_config(settings)
    print(f"Downloading MMAudio {cfg.model_name} into {settings.audio_weights_dir} (progress bars below)...")
    cfg.download_if_needed()
    # The CLIP encoder (apple/DFN5B-CLIP-ViT-H-14-384) comes from the Hugging Face hub; building the feature
    # extractor fetches it too.
    from mmaudio.model.utils.features_utils import FeaturesUtils

    FeaturesUtils(
        tod_vae_ckpt=cfg.vae_path,
        synchformer_ckpt=cfg.synchformer_ckpt,
        enable_conditions=True,
        mode=cfg.mode,
        bigvgan_vocoder_ckpt=cfg.bigvgan_16k_path,
        need_vae_encoder=False,
    )
    size_gb = sum(f.stat().st_size for f in settings.audio_weights_dir.expanduser().glob("*") if f.is_file()) / 1e9
    print(f"Done: {settings.audio_weights_dir} ({size_gb:.1f} GB, plus encoders in ~/.cache/huggingface)")


def download_hunyuan_foley() -> None:
    settings = get_settings()
    target = settings.hunyuan_foley_weights_dir.expanduser()
    print(f"Downloading HunyuanVideo-Foley XL into {target} (progress bars below)...")
    snapshot_download(
        "tencent/HunyuanVideo-Foley",
        local_dir=target,
        allow_patterns=["hunyuanvideo_foley_xl.pth", "synchformer_state_dict.pth", "vae_128d_48k.pth", "*.yaml"],
    )
    # The visual and text encoders it loads by name at run time.
    snapshot_download("google/siglip2-base-patch16-512", allow_patterns=["*.json", "*.model", "*.safetensors"])
    snapshot_download("laion/larger_clap_general", allow_patterns=["*.json", "*.txt", "pytorch_model.bin"])
    print(f"Done: {target}, plus encoders in ~/.cache/huggingface")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0].startswith("--audio="):
        backend = args[0].removeprefix("--audio=")
        if backend == "mmaudio":
            download_mmaudio()
        elif backend == "hunyuan-foley":
            download_hunyuan_foley()
        elif backend == "stable-audio":
            download_hf_model(get_settings().stable_audio_model_id)
        else:
            sys.exit(f"unknown audio backend {backend!r}")
    else:
        download_hf_model(args[0] if args else get_settings().model_id)


if __name__ == "__main__":
    main()
