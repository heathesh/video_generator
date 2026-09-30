# CLAUDE.md

## Project

A local FastAPI service that generates videos on Apple Silicon Macs (developed on an M1 Pro, 32 GB) with the
Hugging Face model **LTX-Video 0.9.5** (`Lightricks/LTX-Video-0.9.5`, 2B params) through 🤗 diffusers'
`LTXConditionPipeline`. It supports text-to-video, and image-to-video when an image is uploaded and used as frame 0.
Everything runs in-process on PyTorch's MPS backend, with no external apps or cloud APIs.

## Commands

The Makefile is the canonical interface (`make help` lists everything):

| | |
|---|---|
| `make setup` | Installs uv via Homebrew if missing, then `uv sync` |
| `make install` | `uv sync` (Python 3.12 is pinned in `.python-version`) |
| `make download-model` | `scripts/download_model.py`: pre-fetches ~25 GB of weights into `~/.cache/huggingface` |
| `make run` / `make dev` | `uv run uvicorn app.main:app` (dev adds `--reload`) |
| `make test` | `uv run pytest` |
| `make health`, `make example-text`, `make example-image IMAGE=...` | curl helpers against a running server |

Always use `uv run` / `uv add`, never bare `pip`. Dev dependencies live in `[dependency-groups] dev`.

## Architecture

```
app/config.py     Settings (pydantic-settings, env prefix VIDEO_, .env supported). Defaults + request limits.
app/generator.py  VideoParams dataclass, snap helpers, Generator protocol, LTXVideoGenerator (the only code touching torch/diffusers).
app/jobs.py       Job dataclass + JobManager: in-memory job dict, asyncio.Queue, single background worker.
app/main.py       create_app(settings, generator) factory + module-level `app`. Routes: /health, POST /generate, /jobs/{id}, /jobs/{id}/video.
scripts/download_model.py  snapshot_download of the configured model.
tests/test_api.py API tests using a FakeGenerator injected via create_app().
```

Request flow: `POST /generate` (multipart form) → validate and snap params → `JobManager.submit` saves the optional
image to `outputs/<job_id>/input.png` and enqueues the job → the worker calls `generator.generate` via
`asyncio.to_thread` → writes `outputs/<job_id>/video.mp4` → clients poll `GET /jobs/{id}` and download the video.

## Conventions and constraints

- **Single serial worker.** One model on one GPU, so never run generations concurrently. All job state is mutated
  on the event loop, and only blocking model calls go to a thread.
- **The model loads in the worker, not in the lifespan.** The server responds right away and `/health` reports
  `loading` → `ok` (or `error` with `load_error`).
- **LTX-Video constraints**, enforced by `snap_dimension` / `snap_num_frames` in `app/generator.py`: width and height
  are multiples of 32, and `num_frames` is `8k + 1`. The snapped values are what's stored in the job's params.
- **torch/diffusers imports stay inside `LTXVideoGenerator` methods** so the API and tests import quickly.
- **Tests must never load the real model.** Inject a fake through `create_app(Settings(output_dir=tmp_path), fake)`.
  Anything that implements the `Generator` protocol works.
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
