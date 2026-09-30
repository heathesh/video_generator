"""Pre-download the model weights into the Hugging Face cache (~/.cache/huggingface) so the first
API start doesn't block on a multi-GB download.

Usage: uv run python scripts/download_model.py [MODEL_ID]
"""

import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402


def main() -> None:
    model_id = sys.argv[1] if len(sys.argv) > 1 else get_settings().model_id
    print(f"Downloading {model_id} (this can take a while; progress bars below)...")
    path = snapshot_download(
        model_id,
        # Weights, configs and tokenizer only; skip example media and single-file checkpoints.
        allow_patterns=["*.json", "*.txt", "*.model", "*/*.safetensors"],
    )
    size_gb = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e9
    print(f"Done: {path} ({size_gb:.1f} GB)")


if __name__ == "__main__":
    main()
