import subprocess
import wave
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
from imageio_ffmpeg import get_ffmpeg_exe

from app.audio import HunyuanFoleyBackend, fit_length, mux_audio, read_wav, write_wav
from app.config import Settings
from app.generator import VideoParams


def sine(seconds: float, rate: int, channels: int = 2) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return np.tile(0.5 * np.sin(2 * np.pi * 440 * t), (channels, 1))


def test_write_wav(tmp_path: Path):
    path = tmp_path / "a.wav"
    write_wav(sine(0.5, 44100), 44100, path)
    with wave.open(str(path)) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (2, 2, 44100, 22050)


def test_write_wav_clips_out_of_range(tmp_path: Path):
    path = tmp_path / "a.wav"
    write_wav(np.array([[2.0, -2.0]]), 8000, path)
    with wave.open(str(path)) as wav:
        assert np.frombuffer(wav.readframes(2), "<i2").tolist() == [32767, -32767]


def test_mux_audio_adds_an_audio_stream(tmp_path: Path):
    video, audio, out = tmp_path / "v.mp4", tmp_path / "a.wav", tmp_path / "out.mp4"
    iio.imwrite(video, np.zeros((24, 64, 64, 3), dtype=np.uint8), fps=24)
    write_wav(sine(1.0, 44100), 44100, audio)

    mux_audio(video, audio, out)

    # `ffmpeg -i` with no output file prints the stream info and exits non-zero.
    probe = subprocess.run([get_ffmpeg_exe(), "-i", str(out)], capture_output=True, text=True, check=False).stderr
    assert "Audio: aac" in probe
    assert "Video: h264" in probe


def test_fit_length_pads_and_trims():
    audio = np.ones((2, 5))
    assert fit_length(audio, 3).shape == (2, 3)
    padded = fit_length(audio, 8)
    assert padded.shape == (2, 8)
    assert padded[:, 5:].sum() == 0


def test_read_wav_roundtrip(tmp_path: Path):
    path = tmp_path / "a.wav"
    original = sine(0.1, 8000)
    write_wav(original, 8000, path)
    samples, rate = read_wav(path)
    assert rate == 8000
    assert samples.shape == original.shape
    assert np.abs(samples - original).max() < 1e-4


def audio_params(**overrides) -> VideoParams:
    values = {
        "prompt": "waves", "negative_prompt": "", "width": 64, "height": 64, "num_frames": 49,
        "num_inference_steps": 1, "guidance_scale": 1.0, "seed": 7, "fps": 24, "audio": True,
        "audio_num_inference_steps": 50,
    }
    return VideoParams(**{**values, **overrides})


def test_hunyuan_command(tmp_path: Path):
    settings = Settings(audio_backend="hunyuan-foley", hunyuan_foley_dir=tmp_path, hunyuan_foley_weights_dir=tmp_path)
    cmd = HunyuanFoleyBackend(settings, "mps").command(audio_params(audio_prompt="surf"), Path("v.mp4"), Path("o.wav"))
    assert cmd[:2] == [str(tmp_path / ".venv" / "bin" / "python"), str(tmp_path / "run.py")]
    args = dict(zip(cmd[2::2], cmd[3::2], strict=False))
    assert args["--prompt"] == "surf"
    assert (args["--seed"], args["--steps"], args["--guidance-scale"], args["--device"]) == ("7", "50", "4.5", "mps")
    assert cmd[-1] == "--offload"


def test_hunyuan_not_installed(tmp_path: Path):
    backend = HunyuanFoleyBackend(Settings(audio_backend="hunyuan-foley", hunyuan_foley_dir=tmp_path), "mps")
    with pytest.raises(RuntimeError, match="setup-audio"):
        backend.load()
