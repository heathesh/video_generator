"""Generate a soundtrack for one video with HunyuanVideo-Foley and write it as a 16-bit wav.

Runs inside this directory's own .venv (see pyproject.toml); app/audio.py calls it as a subprocess. It imports the
upstream code from ./src instead of running its infer.py, because infer.py saves through torchaudio (which needs
torchcodec and FFmpeg's shared libraries) and merges with a system ffmpeg.
"""

import argparse
import sys
import wave
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from hunyuanvideo_foley.utils.model_utils import load_model  # noqa: E402
from infer import infer, set_manual_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True, help="wav file to write")
    parser.add_argument("--weights", required=True, help="directory with hunyuanvideo_foley_xl.pth etc.")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--offload", action="store_true", help="load sub-models on demand to lower peak memory")
    args = parser.parse_args()

    model_dict, cfg = load_model(
        args.weights,
        str(SRC / "configs" / "hunyuanvideo-foley-xl.yaml"),
        torch.device(args.device),
        enable_offload=args.offload,
        model_size="xl",
    )
    set_manual_seed(args.seed)
    audio, sample_rate = infer(
        args.video,
        args.prompt,
        model_dict,
        cfg,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.steps,
        neg_prompt=args.negative_prompt or None,
    )

    pcm = (np.clip(audio.float().cpu().numpy(), -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(args.output, "wb") as wav:
        wav.setnchannels(pcm.shape[0])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.T.tobytes())


if __name__ == "__main__":
    main()
