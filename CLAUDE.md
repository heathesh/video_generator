# CLAUDE.md

## Project

A local FastAPI service that generates videos on Apple Silicon Macs (developed on an M1 Pro, 32 GB) with the
Hugging Face model **LTX-Video 0.9.5** (`Lightricks/LTX-Video-0.9.5`, 2B params) through 🤗 diffusers'
`LTXConditionPipeline`. It supports text-to-video, and image-to-video when an image is uploaded and used as frame 0.
With `audio=true`, an optional audio backend chosen at setup (`VIDEO_AUDIO_BACKEND`: `mmaudio` or `hunyuan-foley` for
synced video-to-audio, `stable-audio` for prompt-only text-to-audio) then adds a soundtrack.
Everything runs in-process on PyTorch's MPS backend, with no external apps or cloud APIs.

## Commands

The Makefile is the canonical interface (`make help` lists everything):

| | |
|---|---|
| `make setup` | Installs uv via Homebrew if missing, then `uv sync` |
| `make install` | `uv sync`, plus `--extra mmaudio` if `.env` selects it (Python 3.12 is pinned in `.python-version`) |
| `make download-model` | `scripts/download_model.py`: pre-fetches ~25 GB of weights into `~/.cache/huggingface` |
| `make setup-audio [AUDIO=...]` | `scripts/setup_audio.py`: interactive chooser. Accept the licence, install the extra, download, write `.env` |
| `make run` / `make dev` | `uv run uvicorn app.main:app` (dev adds `--reload`) |
| `make test` | `uv run pytest` |
| `make health`, `make example-text`, `make example-image IMAGE=...`, `make example-audio` | curl helpers against a running server |

Always use `uv run` / `uv add`, never bare `pip`. Dev dependencies live in `[dependency-groups] dev`.

## Architecture

```
app/config.py     Settings (pydantic-settings, env prefix VIDEO_, .env supported). Defaults + request limits.
app/config.py     also AUDIO_BACKENDS: per-backend max clip length and default steps/guidance (no backend imports).
app/generator.py  VideoParams dataclass, snap helpers, Generator protocol, LTXVideoGenerator (torch/diffusers).
app/audio.py      AudioBackend protocol, StableAudioBackend, MMAudioBackend, HunyuanFoleyBackend (subprocess),
                  make_audio_backend, wav/mux helpers.
backends/hunyuan_foley/  Separate uv project (own pyproject/lock/.venv) + run.py + mps.patch; src/ is a gitignored clone.
app/jobs.py       Job dataclass + JobManager: in-memory job dict, asyncio.Queue, single background worker.
app/main.py       create_app(settings, generator) factory + module-level `app`. Routes: /health, POST /generate, /jobs/{id}, /jobs/{id}/video.
scripts/download_model.py  snapshot_download of the configured model; --audio=<backend> fetches audio weights.
scripts/setup_audio.py     The audio chooser (options table, licence prompt, install, download, .env update).
tests/            test_api.py (FakeGenerator via create_app), test_audio.py (wav/mux helpers), test_setup_audio.py.
```

Request flow: `POST /generate` (multipart form) → validate and snap params → `JobManager.submit` saves the optional
image to `outputs/<job_id>/input.png` and enqueues the job → the worker calls `generator.generate` via
`asyncio.to_thread` → writes `outputs/<job_id>/video.mp4` → if `params.audio`, the configured backend's `add_audio`
generates audio (MMAudio reads that mp4's frames back), writes `audio.wav` and muxes it in with the ffmpeg bundled by
`imageio-ffmpeg` (video stream copied, not re-encoded) → clients poll `GET /jobs/{id}` and
download the video.

## Conventions and constraints

- **Single serial worker.** One model on one GPU, so never run generations concurrently. All job state is mutated
  on the event loop, and only blocking model calls go to a thread.
- **The model loads in the worker, not in the lifespan.** The server responds right away and `/health` reports
  `loading` → `ok` (or `error` with `load_error`).
- **LTX-Video constraints**, enforced by `snap_dimension` / `snap_num_frames` in `app/generator.py`: width and height
  are multiples of 32, and `num_frames` is `8k + 1`. The snapped values are what's stored in the job's params.
- **torch/diffusers/mmaudio imports stay inside methods** (`app/generator.py`, `app/audio.py`) so the API and tests
  import quickly. `app/audio.py` imports `VideoParams` only under `TYPE_CHECKING` to avoid a cycle.
- **Audio backends are optional and chosen at setup.** `audio_backend="none"` is the default, and `audio=true` then
  returns 422. The backend loads lazily on the first audio job. Request limits and defaults come from
  `AUDIO_BACKENDS` in `app/config.py`, so validation never imports a backend. Adding a backend means an entry in
  `AUDIO_BACKENDS`, a class in `app/audio.py`, and an `Option` in `scripts/setup_audio.py`.
- **Each backend installs only its own dependencies.** MMAudio is the `mmaudio` extra (a git dependency pinned to a
  commit). `[tool.uv] override-dependencies` lifts its `numpy<2.1` pin and drops gradio/tensorboard, which only its
  demo and training code use. `uv sync` is exact and would remove the extra, so `make install` passes it from `.env`;
  `uv run` is inexact and leaves it alone. Stable Audio needs nothing extra (diffusers), but its HF repo is gated.
- **HunyuanVideo-Foley can't share the app's env** (numpy 1.26, a transformers fork), so it's a separate uv project in
  `backends/hunyuan_foley/`, cloned at the commit pinned in `scripts/setup_audio.py` and patched with `mps.patch`
  (upstream calls `.cuda()` in its Synchformer path). The app runs `run.py` per job as a subprocess, which writes a
  wav with stdlib `wave` (upstream's `torchaudio.save` needs torchcodec + FFmpeg dylibs). Peak ~14 GB with offload.
- MMAudio weights live in `VIDEO_AUDIO_WEIGHTS_DIR` (`~/.cache/mmaudio`), not its cwd-relative `./weights`. It samples
  noise on the device, so its seed generator is a device generator, not a CPU one. Weights are CC-BY-NC.
- **Tests must never load the real model or import an audio backend.** Inject a fake through
  `create_app(Settings(output_dir=tmp_path, audio_backend=...), fake)`. Anything that implements the `Generator`
  protocol works. Always pass `audio_backend` explicitly so a developer's `.env` can't change the results.
- MPS: `PYTORCH_ENABLE_MPS_FALLBACK=1` is set in `app/generator.py`. Default dtype is bfloat16, and
  `VIDEO_DTYPE=float32` is the escape hatch. VAE tiling is on to limit peak memory, and `torch.mps.empty_cache()` runs
  after each job.
- `sentencepiece` **and** `protobuf` are required. Without protobuf, transformers can't read the T5 `spiece.model` and
  fails with a misleading "`tiktoken` is required" error.
- Seeds use a CPU `torch.Generator` so results are reproducible regardless of device.
- Image uploads: PNG/JPEG/WebP only, validated with Pillow, size limited by `VIDEO_MAX_UPLOAD_MB`, center-cropped
  to the target size with `fit_image`.
- Jobs are in memory only and are lost on restart. `outputs/` is gitignored.
- When changing endpoints, parameters, defaults or make targets, update README.md to match.
