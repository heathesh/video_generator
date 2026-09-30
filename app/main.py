import asyncio
import contextlib
import io
import logging
import random
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from app.config import Settings, get_settings
from app.generator import Generator, LTXVideoGenerator, VideoParams, snap_dimension, snap_num_frames
from app.jobs import JobManager, JobStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP"}


def create_app(settings: Settings | None = None, generator: Generator | None = None) -> FastAPI:
    settings = settings or get_settings()
    video_generator: Generator = generator or LTXVideoGenerator(settings)
    audio_info = settings.audio
    default_audio_steps = audio_info.num_inference_steps if audio_info else VideoParams.audio_num_inference_steps

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        manager = JobManager(video_generator, settings.output_dir)
        app.state.jobs = manager
        # The worker loads the model first, so the API is reachable (and /health reports progress) while it loads.
        worker = asyncio.create_task(manager.run_worker())
        yield
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker

    app = FastAPI(
        title="Local Video Generator",
        description="Text-to-video and image-to-video on Apple Silicon with LTX-Video.",
        version="0.1.0",
        lifespan=lifespan,
    )

    def get_manager() -> JobManager:
        return app.state.jobs

    @app.get("/health")
    def health() -> dict:
        manager = get_manager()
        return {
            "status": "error" if manager.load_error else ("ok" if video_generator.loaded else "loading"),
            "model_id": video_generator.model_id,
            "device": video_generator.device,
            "model_loaded": video_generator.loaded,
            "load_error": manager.load_error,
            "queued_jobs": manager.pending,
            "audio_backend": settings.audio_backend,
        }

    @app.post("/generate", status_code=status.HTTP_202_ACCEPTED)
    async def generate(
        prompt: Annotated[str, Form(min_length=1, max_length=2000)],
        negative_prompt: Annotated[str | None, Form(max_length=2000)] = None,
        image: Annotated[UploadFile | None, File(description="Optional first frame (PNG, JPEG or WebP)")] = None,
        width: Annotated[int | None, Form(ge=64, le=settings.max_width)] = None,
        height: Annotated[int | None, Form(ge=64, le=settings.max_height)] = None,
        num_frames: Annotated[int | None, Form(ge=9, le=settings.max_num_frames)] = None,
        num_inference_steps: Annotated[int | None, Form(ge=1, le=settings.max_num_inference_steps)] = None,
        guidance_scale: Annotated[float | None, Form(ge=1.0, le=20.0)] = None,
        seed: Annotated[int | None, Form(ge=0, le=2**32 - 1)] = None,
        fps: Annotated[int | None, Form(ge=1, le=60)] = None,
        audio: Annotated[bool, Form(description="Add a soundtrack with the configured audio backend")] = False,
        audio_prompt: Annotated[str | None, Form(max_length=2000)] = None,
        audio_num_inference_steps: Annotated[int | None, Form(ge=1, le=settings.max_num_inference_steps)] = None,
    ) -> dict:
        if not prompt.strip():
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "prompt must not be blank")

        pil_image = await _read_image(image, settings.max_upload_mb) if image is not None else None

        params = VideoParams(
            prompt=prompt.strip(),
            negative_prompt=negative_prompt if negative_prompt is not None else settings.default_negative_prompt,
            width=snap_dimension(width or settings.default_width),
            height=snap_dimension(height or settings.default_height),
            num_frames=snap_num_frames(num_frames or settings.default_num_frames),
            num_inference_steps=num_inference_steps or settings.default_num_inference_steps,
            guidance_scale=guidance_scale if guidance_scale is not None else settings.default_guidance_scale,
            seed=seed if seed is not None else random.randint(0, 2**32 - 1),
            fps=fps or settings.default_fps,
            audio=audio,
            audio_prompt=audio_prompt.strip() or None if audio_prompt else None,
            audio_num_inference_steps=audio_num_inference_steps or default_audio_steps,
        )
        if audio:
            if audio_info is None:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT, "audio isn't set up on this server (run make setup-audio)"
                )
            if params.num_frames / params.fps > audio_info.max_seconds:
                raise HTTPException(
                    status.HTTP_422_UNPROCESSABLE_CONTENT,
                    f"video is {params.num_frames / params.fps:.1f} s; {settings.audio_backend} audio supports at "
                    f"most {audio_info.max_seconds:g} s (lower num_frames or raise fps)",
                )
        manager = get_manager()
        job = manager.submit(params, pil_image)
        return job.to_dict(manager.queue_position(job))

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        manager = get_manager()
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
        return job.to_dict(manager.queue_position(job))

    @app.get("/jobs/{job_id}/video", response_class=FileResponse)
    def get_video(job_id: str) -> FileResponse:
        job = get_manager().get(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
        if job.status != JobStatus.COMPLETED:
            raise HTTPException(status.HTTP_409_CONFLICT, f"job is {job.status}, video not available")
        return FileResponse(job.video_path, media_type="video/mp4", filename=f"{job_id}.mp4")

    return app


async def _read_image(upload: UploadFile, max_upload_mb: int) -> Image.Image:
    data = await upload.read(max_upload_mb * 1024 * 1024 + 1)
    if len(data) > max_upload_mb * 1024 * 1024:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"image must be at most {max_upload_mb} MB")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "image is not a valid PNG, JPEG or WebP file") from exc
    if image.format not in ALLOWED_IMAGE_FORMATS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"unsupported image format {image.format}")
    return image.convert("RGB")


app = create_app()
