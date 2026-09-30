# video_generator

A local HTTP API that generates videos on an Apple Silicon Mac (M1 or later) using the open
[LTX-Video](https://huggingface.co/Lightricks/LTX-Video-0.9.5) model from Hugging Face. There's no third-party desktop
app and no cloud service: the model runs in Python on your Mac's GPU through PyTorch's Metal (MPS) backend.

- **Text-to-video**: `POST` a prompt and get an MP4 back.
- **Image-to-video**: `POST` a prompt plus an image, and the image becomes the first frame of the video.
- **Audio** (optional): add `audio=true` to get a soundtrack in the MP4. You choose the audio model once with
  `make setup-audio`, trading off sync to the video against licence terms. See
  [Choosing an audio option](#choosing-an-audio-option).

Built and tested on a MacBook M1 Pro with 32 GB of RAM.

---

## Requirements

| | |
|---|---|
| Mac | Apple Silicon (M1/M2/M3/M4). 32 GB of unified memory recommended; 16 GB may work with small resolutions and frame counts. |
| macOS | 13 (Ventura) or newer |
| Disk | ~27 GB free for the video model and dependencies, 5–11 GB more for an optional audio model, plus room for generated videos |
| Tools | [Homebrew](https://brew.sh), Xcode Command Line Tools (`xcode-select --install`), `make` and `curl` (both ship with macOS) |

You don't need to install Python yourself. [uv](https://docs.astral.sh/uv/) downloads the right version (3.12) into
the project.

---

## Quick start

> [!IMPORTANT]
> **The first-time setup downloads up to ~37 GB and can take hours.** Besides the Python packages, it downloads
> the AI models themselves. This only happens once, since everything is cached afterwards. Rough times:
>
> | Download | Size | ~10 MB/s (80 Mbps) | ~3 MB/s (25 Mbps) |
> |---|---|---|---|
> | Python dependencies (`make setup`) | ~1.7 GB installed | ~3 min | ~10 min |
> | Video model (`make download-model`) | ~25 GB | ~45 min | **~2–2.5 hours** |
> | Audio model, optional (`make setup-audio`) | 5.3–10.7 GB, [depends on the option](#choosing-an-audio-option) | ~9–18 min | **~30–60 min** |
>
> The ~3 MB/s column matches a real first install on a home connection. Anonymous Hugging Face downloads can be
> rate-limited, so log in first (see step 4 [below](#step-by-step-without-make)) to get the full speed of your
> connection. If a download is interrupted, re-run the same command: the video model resumes where it stopped, and
> the audio model keeps the files that finished and re-downloads only the one that was cut off.

```bash
git clone <this repo> video_generator
cd video_generator

make setup                  # 1. installs uv (via Homebrew) if missing, then Python 3.12 + all dependencies
make download-model         # 2. downloads the video model weights (~25 GB, one time only)
make setup-audio            #    optional: choose and download an audio model, only needed for audio=true
make run                    # 3. starts the API on http://127.0.0.1:8000
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
| `make install` | `uv sync`: installs Python 3.12 and all dependencies into `.venv`, keeping the chosen audio option's |
| `make download-model` | Pre-downloads the model weights (~25 GB) |
| `make setup-audio` | Chooses, installs and downloads an audio model, and saves the choice to `.env`. Interactive, or `AUDIO=mmaudio` / `hunyuan-foley` / `stable-audio` / `none` |
| `make run` | Starts the API server. Override the address with `HOST=0.0.0.0 PORT=9000` |
| `make dev` | Same as `run`, but restarts on code changes (and reloads the model each time) |
| `make test` | Runs the test suite. It uses a fake model, so it's fast and needs no download |
| `make health` | `GET /health` against the running server |
| `make example-text` | Submits a text-to-video job. Customize it with `PROMPT="..."` |
| `make example-image` | Submits an image-to-video job: `IMAGE=path/to.png PROMPT="..."` |
| `make example-audio` | Submits a text-to-video job with audio (run `make setup-audio` first). Customize it with `PROMPT="..."` |
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

### Adding audio

Once an audio option is set up (see below), add `audio=true` to either mode. After the video is generated, the audio
model writes a soundtrack and the MP4 you download has it built in. It also stays next to the video as
`outputs/<job_id>/audio.wav`:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -F "prompt=A glass falls off a kitchen table and shatters on the tile floor" \
  -F num_frames=97 \
  -F audio=true \
  -F "audio_prompt=glass shattering on tiles"     # optional, defaults to the video prompt
```

If audio isn't set up, `audio=true` returns `422`. Jobs without `audio=true` never load the audio model.

### Choosing an audio option

Run `make setup-audio`. It shows the options below, asks you to pick one and accept its licence, installs what that
option needs, downloads its weights and saves the choice as `VIDEO_AUDIO_BACKEND` in `.env`. Restart the server
afterwards. Run it again at any time to switch, or pick `none` to turn audio off. For scripts, use
`make setup-audio AUDIO=mmaudio` (it still asks you to accept the licence).

| Option | Synced to the video? | Commercial use | Max clip | Download |
|---|---|---|---|---|
| `mmaudio`: [MMAudio](https://github.com/hkchengrex/MMAudio) | **Yes.** It watches the frames, so a glass shatters on the impact frame | **No**, [CC-BY-NC 4.0](https://huggingface.co/hkchengrex/MMAudio) | 10 s | ~10.7 GB |
| `hunyuan-foley`: [HunyuanVideo-Foley XL](https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley) | **Yes.** It watches the frames too, with 48 kHz output | **Yes, with limits**: not licensed in the EU, UK or South Korea; over 100M monthly users needs Tencent's permission; shared output must be marked as AI-generated. [Licence](https://huggingface.co/tencent/HunyuanVideo-Foley/blob/main/LICENSE) | 15 s | ~10.6 GB |
| `stable-audio`: [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0) | **No.** It only reads the prompt, so the sound fits the scene but isn't timed to it | Yes, if you earn under US$1M a year. You must register with Stability AI and show "Powered by Stability AI" if you distribute it. [Licence](https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/LICENSE.md) | 47 s | ~5.3 GB |
| LTX-2 *(not available)* | Yes, since it generates video and audio together | Yes, under US$10M revenue | | Needs a Mac with 64 GB+ |

- All options make sound effects and ambience (Stable Audio also music). None produces intelligible speech.
- MMAudio is trained on 8-second clips. The default 2–4 second clips work well.
- **Speed and memory** on an M1 Pro (32 GB), for a 2-second clip:

  | Option | Time added per job | Peak extra memory | Notes |
  |---|---|---|---|
  | `mmaudio` | ~10 s (plus ~45 s to load on the first audio job) | ~5 GB | Stays loaded next to the video model |
  | `hunyuan-foley` | ~50–70 s | ~14 GB | Runs as a separate process per job and frees its memory afterwards. On 32 GB, macOS swaps part of the idle video model out while it runs (about 9 GB of swap in testing). It works, but close other apps |
  | `stable-audio` | not measured yet | | |

- **HunyuanVideo-Foley runs in its own Python environment** (`backends/hunyuan_foley/`), because it needs versions
  of numpy and transformers that clash with the rest of the app. `make setup-audio` clones it at a pinned commit
  into `backends/hunyuan_foley/src`, applies a small patch so it runs on Apple Silicon (upstream hard-codes CUDA in
  one place) and installs that environment.
- **Stable Audio is gated on Hugging Face.** Before `make setup-audio`, open its
  [model page](https://huggingface.co/stabilityai/stable-audio-open-1.0), accept the terms and run
  `uv run hf auth login`. The setup checks this and tells you if it's missing.
- `make install` keeps the chosen option's dependencies installed (it reads `.env`).

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
| `seed` | random | Set it to reproduce a result (also seeds the audio) |
| `audio` | `false` | `true` adds a soundtrack. Needs `make setup-audio`, and the clip must fit the option's max length (see above) |
| `audio_prompt` | the `prompt` | Describe the sounds you want |
| `audio_num_inference_steps` | `25` (mmaudio), `50` (hunyuan-foley), `100` (stable-audio) | Audio sampling steps |

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
| `VIDEO_AUDIO_BACKEND` | `none` | `mmaudio`, `hunyuan-foley`, `stable-audio` or `none`. Set by `make setup-audio` |
| `VIDEO_AUDIO_DTYPE` | `bfloat16` | Try `float32` if the audio comes out silent or as noise |
| `VIDEO_DEFAULT_AUDIO_NUM_INFERENCE_STEPS` / `_AUDIO_GUIDANCE_SCALE` / `_AUDIO_NEGATIVE_PROMPT` | the option's own (mmaudio 25 / 4.5, hunyuan-foley 50 / 4.5, stable-audio 100 / 7) / empty | Audio defaults |
| `VIDEO_AUDIO_VARIANT` | `large_44k_v2` | MMAudio model size: `small_16k`, `small_44k`, `medium_44k`, `large_44k`, `large_44k_v2` |
| `VIDEO_AUDIO_WEIGHTS_DIR` | `~/.cache/mmaudio` | Where the MMAudio weights are stored |
| `VIDEO_HUNYUAN_FOLEY_WEIGHTS_DIR` | `~/.cache/hunyuan-foley` | Where the HunyuanVideo-Foley weights are stored |
| `VIDEO_HUNYUAN_FOLEY_OFFLOAD` | `true` | Load its sub-models on demand to lower peak memory |
| `VIDEO_STABLE_AUDIO_MODEL_ID` | `stabilityai/stable-audio-open-1.0` | The Stable Audio repo |

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
| `audio=true` returns 422 "audio isn't set up" | Run `make setup-audio` and restart the server |
| Audio is silent, noise, or the job fails in the audio step | Set `VIDEO_AUDIO_DTYPE=float32` and restart |
| A `hunyuan-foley` job fails with "isn't installed" | Run `make setup-audio AUDIO=hunyuan-foley` again |
| `make setup-audio` says it can't download Stable Audio | Accept the terms on its [model page](https://huggingface.co/stabilityai/stable-audio-open-1.0) with the account you logged in with (`uv run hf auth login`) |
| Stop the server | `Ctrl+C` in its terminal. Queued jobs live in memory and are lost on restart; finished videos stay in `outputs/` |

## Development

```bash
make test   # pytest with a fake generator, no model or GPU needed
make dev    # auto-reload server
```

See [CLAUDE.md](CLAUDE.md) for the architecture and conventions.

## License

MIT for this code (see [LICENSE](LICENSE)). The LTX-Video weights have their own license on the
[model page](https://huggingface.co/Lightricks/LTX-Video-0.9.5); review it before any commercial use. Each audio
option has its own licence (see [Choosing an audio option](#choosing-an-audio-option)). In particular, audio made
with `mmaudio` is for non-commercial use only.
