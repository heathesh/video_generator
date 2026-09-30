# video_generator

A local HTTP API that generates videos on an Apple Silicon Mac (M1 or later) using the open
[LTX-Video](https://huggingface.co/Lightricks/LTX-Video-0.9.5) model from Hugging Face. There's no third-party desktop
app and no cloud service: the model runs in Python on your Mac's GPU through PyTorch's Metal (MPS) backend.

- **Text-to-video**: `POST` a prompt and get an MP4 back.
- **Image-to-video**: `POST` a prompt plus an image, and the image becomes the first frame of the video.

Built and tested on a MacBook M1 Pro with 32 GB of RAM.

---

## Requirements

| | |
|---|---|
| Mac | Apple Silicon (M1/M2/M3/M4). 32 GB of unified memory recommended; 16 GB may work with small resolutions and frame counts. |
| macOS | 13 (Ventura) or newer |
| Disk | ~26 GB free for the model weights, plus room for generated videos |
| Tools | [Homebrew](https://brew.sh), Xcode Command Line Tools (`xcode-select --install`), `make` and `curl` (both ship with macOS) |

You don't need to install Python yourself. [uv](https://docs.astral.sh/uv/) downloads the right version (3.12) into
the project.

---

## Quick start

```bash
git clone <this repo> video_generator
cd video_generator

make setup            # 1. installs uv (via Homebrew) if missing, then Python 3.12 + all dependencies
make download-model   # 2. downloads the model weights (~25 GB, one time only)
make run              # 3. starts the API on http://127.0.0.1:8000
```

Then, in a second terminal:

```bash
make health                                                    # wait until "model_loaded": true
make example-text PROMPT="A paper boat drifting down a rainy street"
make example-image IMAGE=~/Pictures/dog.jpg PROMPT="The dog turns its head and wags its tail"
```

Each `example-*` command prints a `job_id`. See [Using the API](#using-the-api) to check on the job and download the
video.

### Step by step without `make`

Each `make` target is a thin wrapper, so you can run the commands yourself.

1. **Install Homebrew** (skip if `brew --version` works):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. **Install uv**, the Python package and project manager:
   ```bash
   brew install uv
   ```
3. **Install Python 3.12 and the dependencies** into `./.venv` (PyTorch, diffusers, transformers, FastAPI, ...):
   ```bash
   uv sync
   ```
4. **Download the model** into `~/.cache/huggingface`. This is optional, because the server otherwise downloads it
   on first start, but it lets you watch the progress:
   ```bash
   uv run python scripts/download_model.py
   ```
   The model is public, so you don't need a Hugging Face account. Anonymous downloads are rate-limited, though, so
   logging in first is recommended: create a free **Read** token at https://huggingface.co/settings/tokens and run
   `uv run hf auth login`. Expect the download to take a while. It took about 2 hours on a home connection
   (~3 MB/s average).
5. **Start the server**:
   ```bash
   uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```
   The API starts at once and loads the model in the background, which takes about 30 seconds. `GET /health` shows
   `"status": "loading"` until the model is ready. Jobs submitted before then wait in the queue.

### Makefile targets

| Target | What it does |
|---|---|
| `make help` | Lists all targets (also the default when you run `make`) |
| `make setup` | Checks the Mac is Apple Silicon, installs `uv` via Homebrew if missing, runs `make install` |
| `make install` | `uv sync`: installs Python 3.12 and all dependencies into `.venv` |
| `make download-model` | Pre-downloads the model weights (~25 GB) |
| `make run` | Starts the API server. Override the address with `HOST=0.0.0.0 PORT=9000` |
| `make dev` | Same as `run`, but restarts on code changes (and reloads the model each time) |
| `make test` | Runs the test suite. It uses a fake model, so it's fast and needs no download |
| `make health` | `GET /health` against the running server |
| `make example-text` | Submits a text-to-video job. Customize it with `PROMPT="..."` |
| `make example-image` | Submits an image-to-video job: `IMAGE=path/to.png PROMPT="..."` |
| `make clean` | Deletes `outputs/`, `.venv` and test caches. The model cache in `~/.cache/huggingface` is kept |

---

## Using the API

Generating a video takes minutes, so the API is asynchronous:

1. `POST /generate` queues a job and immediately returns `202 Accepted` with a `job_id`.
2. `GET /jobs/{job_id}` reports `queued` → `running` → `completed` (or `failed`).
3. `GET /jobs/{job_id}/video` downloads the MP4 once the job is `completed`.

Jobs run one at a time, in order. Interactive docs are at **http://127.0.0.1:8000/docs**.

### Text-to-video

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -F "prompt=A red fox trotting through fresh snow in a pine forest, soft morning light, cinematic" \
  -F num_frames=49 \
  -F seed=42
```

### Image-to-video

Add an `image` file field (PNG, JPEG or WebP, up to 20 MB). The image is resized and center-cropped to the video size
and used as the first frame. Describe the motion you want in the prompt:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -F "prompt=The woman smiles and turns towards the camera, her hair moving in the breeze" \
  -F "image=@/path/to/photo.jpg" \
  -F width=480 -F height=704     # portrait: match your image's orientation
```

Response (both modes):

```json
{
  "job_id": "3f2c9a...",
  "status": "queued",
  "mode": "image-to-video",
  "params": {"prompt": "...", "width": 480, "height": 704, "num_frames": 97, "seed": 1234567, ...},
  "queue_position": 1,
  "video_url": null,
  ...
}
```

### Check status and download

```bash
curl http://127.0.0.1:8000/jobs/<job_id>
# {"status": "running", "duration_seconds": 84.2, ...}

curl -o video.mp4 http://127.0.0.1:8000/jobs/<job_id>/video
open video.mp4
```

A one-liner that waits and then downloads:

```bash
JOB=$(curl -s -X POST http://127.0.0.1:8000/generate -F "prompt=Waves crashing on black sand at sunset" -F num_frames=49 | python3 -c 'import sys,json;print(json.load(sys.stdin)["job_id"])')
until curl -s http://127.0.0.1:8000/jobs/$JOB | grep -qE '"status":"(completed|failed)"'; do sleep 10; done
curl -o "$JOB.mp4" http://127.0.0.1:8000/jobs/$JOB/video && open "$JOB.mp4"
```

Videos are also saved on disk in `outputs/<job_id>/video.mp4`, next to the uploaded `input.png` for image-to-video
jobs.

### Endpoints

| Method & path | Description |
|---|---|
| `GET /health` | `status` (`loading` / `ok` / `error`), `model_id`, `device` (should be `mps`), `model_loaded`, `queued_jobs` |
| `POST /generate` | `multipart/form-data`, fields below. Returns `202` with the job |
| `GET /jobs/{job_id}` | Job status, parameters, timings, `error` if failed, `video_url` when done |
| `GET /jobs/{job_id}/video` | The MP4. `404` for an unknown job, `409` if it isn't completed |

### `POST /generate` fields

| Field | Default | Notes |
|---|---|---|
| `prompt` | *(required)* | Describe the scene **and the motion**. Long, detailed, chronological prompts work best with LTX-Video |
| `image` | none | Optional PNG/JPEG/WebP file. If present, it becomes the first frame (image-to-video) |
| `negative_prompt` | `worst quality, inconsistent motion, blurry, jittery, distorted` | Things to avoid |
| `width` / `height` | `704` / `480` | Rounded down to a multiple of 32. Max 1280 |
| `num_frames` | `97` (~4 s) | Rounded down to `8k + 1` (9, 17, 25, ... 257) |
| `fps` | `24` | Playback frame rate; also passed to the model |
| `num_inference_steps` | `40` | More steps: better quality, slower. 20–30 is fine for drafts |
| `guidance_scale` | `3.0` | How closely to follow the prompt (1–20) |
| `seed` | random | Set it to reproduce a result |

The values you actually get back, after rounding, are shown in the job's `params`.

---

## Performance and memory

Measured on a MacBook M1 Pro with 32 GB, with the default 40 steps:

| Job | Resolution | Frames | Time |
|---|---|---|---|
| Text-to-video | 704×480 | 49 (~2 s) | ~3 min 10 s |
| Image-to-video | 512×512 | 49 (~2 s) | ~2 min 50 s |
| Model load at server start | | | ~30 s |

- Time grows roughly in proportion to `width × height × num_frames` and to `num_inference_steps`. A default 97-frame
  clip takes about twice as long as the 49-frame clips above. Use `num_inference_steps=20` for faster drafts.
- The model (LTX-Video 0.9.5, 2B parameters, plus a T5-XXL text encoder) uses roughly **13–15 GB** of unified memory
  in bfloat16, plus working memory that grows with resolution and frame count. About a third of system memory stayed
  free during the runs above.
- For image-to-video, fast motion such as flapping wings tends to blur by the end of a clip. Gentle, continuous
  motion gives the cleanest results.
- Close other memory-heavy apps while generating. If macOS starts swapping heavily, reduce the resolution or frame
  count.

## Configuration

Settings come from environment variables prefixed with `VIDEO_`, or from a `.env` file in the project root:

| Variable | Default | |
|---|---|---|
| `VIDEO_MODEL_ID` | `Lightricks/LTX-Video-0.9.5` | Any diffusers-format LTX-Video repo that works with `LTXConditionPipeline` |
| `VIDEO_DTYPE` | `bfloat16` | Try `float32` if you get black or garbled output (uses about twice the memory) |
| `VIDEO_DEVICE` | `auto` | `mps`, or `cpu` (very slow) |
| `VIDEO_OUTPUT_DIR` | `outputs` | Where videos and uploaded images are stored |
| `VIDEO_DEFAULT_WIDTH` / `_HEIGHT` / `_NUM_FRAMES` / `_NUM_INFERENCE_STEPS` / `_GUIDANCE_SCALE` / `_FPS` / `_NEGATIVE_PROMPT` | see above | Request defaults |
| `VIDEO_MAX_WIDTH` / `_HEIGHT` / `_NUM_FRAMES` / `_NUM_INFERENCE_STEPS` / `_UPLOAD_MB` | 1280 / 1280 / 257 / 100 / 20 | Request limits |

Example: `VIDEO_DEFAULT_NUM_FRAMES=49 make run`

## Troubleshooting

| Problem | Fix |
|---|---|
| `/health` says `"device": "cpu"` | You're not on Apple Silicon or are running an x86 (Rosetta) terminal. Check that `uname -m` prints `arm64` |
| `"status": "error"` with a `load_error` | Usually an incomplete download. Re-run `make download-model` (it resumes) and restart |
| Out-of-memory errors, or the Mac freezes and swaps | Lower `width`/`height`/`num_frames`. Close other apps |
| Black, noisy or garbled video | Set `VIDEO_DTYPE=float32` and restart |
| `NotImplementedError ... MPS` | The server already sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so missing ops run on the CPU. Update with `uv sync --upgrade` |
| Image-to-video ignores the image composition | Match `width`/`height` to your image's aspect ratio, since the image is center-cropped to fit |
| Stop the server | `Ctrl+C` in its terminal. Queued jobs live in memory and are lost on restart; finished videos stay in `outputs/` |

## Development

```bash
make test   # pytest with a fake generator, no model or GPU needed
make dev    # auto-reload server
```

See [CLAUDE.md](CLAUDE.md) for the architecture and conventions.

## License

MIT for this code (see [LICENSE](LICENSE)). The LTX-Video weights have their own license on the
[model page](https://huggingface.co/Lightricks/LTX-Video-0.9.5); review it before any commercial use.
