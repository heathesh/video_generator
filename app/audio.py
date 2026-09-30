"""Optional audio step: add a soundtrack to a finished video. The backend is chosen with VIDEO_AUDIO_BACKEND
(normally by `make setup-audio`); each one loads lazily on the first job that asks for audio."""

import gc
import logging
import subprocess
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from app.config import Settings

if TYPE_CHECKING:
    from app.generator import VideoParams

logger = logging.getLogger(__name__)


class AudioBackend(Protocol):
    name: str

    def load(self) -> None: ...

    def add_audio(self, params: "VideoParams", video_path: Path) -> None:
        """Generate audio for video_path and replace it with a copy that has the soundtrack."""
        ...


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM wav as (channels, samples) float audio in [-1, 1], plus its sample rate."""
    with wave.open(str(path), "rb") as wav:
        channels, rate = wav.getnchannels(), wav.getframerate()
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), "<i2")
    return pcm.reshape(-1, channels).T.astype(np.float32) / 32767, rate


def write_wav(samples: np.ndarray, sample_rate: int, path: Path) -> None:
    """Write (channels, samples) float audio in [-1, 1] as 16-bit PCM."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(pcm.shape[0])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.T.tobytes())  # interleave channels


def fit_length(samples: np.ndarray, num_samples: int) -> np.ndarray:
    """Trim (channels, samples) audio to num_samples, or pad the end with silence."""
    if samples.shape[1] >= num_samples:
        return samples[:, :num_samples]
    return np.pad(samples, ((0, 0), (0, num_samples - samples.shape[1])))


def mux_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
    """Copy the video stream (no re-encode) and add audio_path as an AAC track."""
    from imageio_ffmpeg import get_ffmpeg_exe

    subprocess.run(
        [get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video_path), "-i", str(audio_path),
         "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", str(output_path)],
        check=True,
        capture_output=True,
    )


def replace_with_audio(video_path: Path, samples: np.ndarray, sample_rate: int, duration_s: float) -> None:
    """Save the soundtrack next to the video as audio.wav, then swap in a copy of the video that includes it.

    The audio is fitted to exactly duration_s first: backends can return slightly less (MMAudio truncates to its
    sync-feature length), and muxing with -shortest would then cut off the last video frames.
    """
    audio_path = video_path.with_name("audio.wav")
    write_wav(fit_length(samples, round(duration_s * sample_rate)), sample_rate, audio_path)
    with_audio = video_path.with_name(f"{video_path.stem}.audio{video_path.suffix}")
    mux_audio(video_path, audio_path, with_audio)
    with_audio.replace(video_path)


def free_device_memory(device: str) -> None:
    import torch

    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


class StableAudioBackend:
    """Stable Audio Open: text-to-audio from the prompt. It never sees the video, so the sound isn't synced."""

    name = "stable-audio"

    def __init__(self, settings: Settings, device: str):
        self.settings = settings
        self.device = device
        self._pipe = None

    def load(self) -> None:
        import torch
        from diffusers import StableAudioPipeline

        logger.info("Loading %s (%s) on %s ...", self.settings.stable_audio_model_id, self.settings.audio_dtype,
                    self.device)
        pipe = StableAudioPipeline.from_pretrained(
            self.settings.stable_audio_model_id, torch_dtype=getattr(torch, self.settings.audio_dtype)
        )
        self._pipe = pipe.to(self.device)
        logger.info("Stable Audio loaded.")

    def add_audio(self, params: "VideoParams", video_path: Path) -> None:
        import torch

        if self._pipe is None:
            self.load()
        try:
            audio = self._pipe(
                prompt=params.audio_prompt or params.prompt,
                negative_prompt=self.settings.default_audio_negative_prompt or None,
                audio_end_in_s=params.num_frames / params.fps,
                num_inference_steps=params.audio_num_inference_steps,
                guidance_scale=self.settings.audio.guidance_scale,
                generator=torch.Generator(device="cpu").manual_seed(params.seed),
            ).audios[0]
            replace_with_audio(
                video_path, audio.float().cpu().numpy(), self._pipe.vae.sampling_rate, params.num_frames / params.fps
            )
        finally:
            free_device_memory(self.device)


def mmaudio_model_config(settings: Settings):
    """The MMAudio ModelConfig for the configured variant, with weights under audio_weights_dir instead of ./weights."""
    import dataclasses

    from mmaudio.eval_utils import all_model_cfg

    cfg = all_model_cfg[settings.audio_variant]
    root = settings.audio_weights_dir.expanduser()
    return dataclasses.replace(
        cfg,
        model_path=root / cfg.model_path.name,
        vae_path=root / cfg.vae_path.name,
        bigvgan_16k_path=root / cfg.bigvgan_16k_path.name if cfg.bigvgan_16k_path else None,
        synchformer_ckpt=root / cfg.synchformer_ckpt.name,
    )


class MMAudioBackend:
    """MMAudio: watches the frames (and reads the prompt), so the sound lines up with the motion."""

    name = "mmaudio"

    def __init__(self, settings: Settings, device: str):
        self.settings = settings
        self.device = device
        self._cfg = None
        self._net = None
        self._features = None

    def load(self) -> None:
        import torch
        from mmaudio.model.networks import get_my_mmaudio
        from mmaudio.model.utils.features_utils import FeaturesUtils

        dtype = getattr(torch, self.settings.audio_dtype)
        cfg = mmaudio_model_config(self.settings)
        logger.info("Loading MMAudio %s (%s) on %s ...", cfg.model_name, self.settings.audio_dtype, self.device)
        cfg.download_if_needed()
        net = get_my_mmaudio(cfg.model_name).to(self.device, dtype).eval()
        net.load_weights(torch.load(cfg.model_path, map_location=self.device, weights_only=True))
        features = FeaturesUtils(
            tod_vae_ckpt=cfg.vae_path,
            synchformer_ckpt=cfg.synchformer_ckpt,
            enable_conditions=True,
            mode=cfg.mode,
            bigvgan_vocoder_ckpt=cfg.bigvgan_16k_path,
            need_vae_encoder=False,
        )
        self._cfg, self._net, self._features = cfg, net, features.to(self.device, dtype).eval()
        logger.info("MMAudio loaded.")

    def add_audio(self, params: "VideoParams", video_path: Path) -> None:
        import torch
        from mmaudio.eval_utils import generate, load_video
        from mmaudio.model.flow_matching import FlowMatching

        if self._net is None:
            self.load()
        try:
            with torch.inference_mode():
                video = load_video(video_path, params.num_frames / params.fps)
                seq_cfg = self._cfg.seq_cfg
                seq_cfg.duration = video.duration_sec
                self._net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)
                # MMAudio draws its noise on the model's device, so the seeded generator has to live there too.
                rng = torch.Generator(device=self.device).manual_seed(params.seed)
                audio = generate(
                    video.clip_frames.unsqueeze(0),
                    video.sync_frames.unsqueeze(0),
                    [params.audio_prompt or params.prompt],
                    negative_text=[self.settings.default_audio_negative_prompt],
                    feature_utils=self._features,
                    net=self._net,
                    fm=FlowMatching(min_sigma=0, inference_mode="euler", num_steps=params.audio_num_inference_steps),
                    rng=rng,
                    cfg_strength=self.settings.audio.guidance_scale,
                )
            replace_with_audio(
                video_path, audio.float().cpu()[0].numpy(), seq_cfg.sampling_rate, params.num_frames / params.fps
            )
        finally:
            free_device_memory(self.device)


class HunyuanFoleyBackend:
    """HunyuanVideo-Foley XL: watches the frames, so the sound lines up with the motion.

    It needs dependencies that conflict with this app (numpy 1.26, a transformers fork), so it lives in its own
    environment under settings.hunyuan_foley_dir and runs as a subprocess per job. The subprocess also hands all of
    its memory back when it exits.
    """

    name = "hunyuan-foley"

    def __init__(self, settings: Settings, device: str):
        self.settings = settings
        self.device = device
        self.python = settings.hunyuan_foley_dir / ".venv" / "bin" / "python"
        self.runner = settings.hunyuan_foley_dir / "run.py"

    def load(self) -> None:
        # Nothing stays loaded between jobs; just check that `make setup-audio` has installed it.
        if not self.python.exists() or not (self.settings.hunyuan_foley_dir / "src").exists():
            raise RuntimeError("hunyuan-foley isn't installed; run make setup-audio AUDIO=hunyuan-foley")

    def command(self, params: "VideoParams", video_path: Path, output_wav: Path) -> list[str]:
        cmd = [
            str(self.python), str(self.runner),
            "--video", str(video_path),
            "--prompt", params.audio_prompt or params.prompt,
            "--output", str(output_wav),
            "--weights", str(self.settings.hunyuan_foley_weights_dir.expanduser()),
            "--device", self.device,
            "--seed", str(params.seed),
            "--steps", str(params.audio_num_inference_steps),
            "--guidance-scale", str(self.settings.audio.guidance_scale),
            "--negative-prompt", self.settings.default_audio_negative_prompt,
        ]
        if self.settings.hunyuan_foley_offload:
            cmd.append("--offload")
        return cmd

    def add_audio(self, params: "VideoParams", video_path: Path) -> None:
        import os

        self.load()
        free_device_memory(self.device)  # leave as much room as possible for the subprocess
        raw = video_path.with_name("audio.raw.wav")
        env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
        # check=False: a failure is reported below with the end of its stderr, which is where the reason is.
        result = subprocess.run(
            self.command(params, video_path, raw), env=env, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            tail = "\n".join(result.stderr.strip().splitlines()[-15:])
            raise RuntimeError(f"hunyuan-foley failed (exit {result.returncode}):\n{tail}")
        samples, rate = read_wav(raw)
        raw.unlink()
        replace_with_audio(video_path, samples, rate, params.num_frames / params.fps)


def make_audio_backend(settings: Settings, device: str) -> AudioBackend | None:
    """The configured backend (not yet loaded), or None when audio isn't set up."""
    backends: dict[str, type] = {
        "stable-audio": StableAudioBackend,
        "mmaudio": MMAudioBackend,
        "hunyuan-foley": HunyuanFoleyBackend,
    }
    if settings.audio_backend == "none":
        return None
    if settings.audio_backend not in backends:
        raise ValueError(f"audio backend {settings.audio_backend!r} is not available")
    return backends[settings.audio_backend](settings, device)
