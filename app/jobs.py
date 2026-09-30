import asyncio
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path

from PIL import Image

from app.generator import Generator, VideoParams

logger = logging.getLogger(__name__)


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    params: VideoParams
    job_dir: Path
    image_path: Path | None = None
    status: JobStatus = JobStatus.QUEUED
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def mode(self) -> str:
        return "image-to-video" if self.image_path else "text-to-video"

    @property
    def video_path(self) -> Path:
        return self.job_dir / "video.mp4"

    def to_dict(self, queue_position: int | None = None) -> dict:
        duration = None
        if self.started_at is not None:
            duration = round((self.finished_at or time.time()) - self.started_at, 1)
        return {
            "job_id": self.id,
            "status": self.status,
            "mode": self.mode,
            "params": asdict(self.params),
            "queue_position": queue_position,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": duration,
            "error": self.error,
            "video_url": f"/jobs/{self.id}/video" if self.status == JobStatus.COMPLETED else None,
        }


class JobManager:
    """In-memory job store plus a single background worker.

    There is one model on one GPU, so jobs run strictly one at a time. All job state is mutated
    on the event loop; only the blocking model calls run in a worker thread.
    """

    def __init__(self, generator: Generator, output_dir: Path):
        self.generator = generator
        self.output_dir = output_dir
        self.load_error: str | None = None
        self._jobs: dict[str, Job] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    def submit(self, params: VideoParams, image: Image.Image | None) -> Job:
        job_id = uuid.uuid4().hex
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        image_path = None
        if image is not None:
            image_path = job_dir / "input.png"
            image.save(image_path)
        job = Job(id=job_id, params=params, job_dir=job_dir, image_path=image_path)
        self._jobs[job_id] = job
        self._queue.put_nowait(job_id)
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def queue_position(self, job: Job) -> int | None:
        """1-based position among queued jobs, or None if the job isn't waiting."""
        if job.status != JobStatus.QUEUED:
            return None
        queued = [j for j in self._jobs.values() if j.status == JobStatus.QUEUED]
        queued.sort(key=lambda j: j.created_at)
        return queued.index(job) + 1

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    async def run_worker(self) -> None:
        if not self.generator.loaded:
            try:
                await asyncio.to_thread(self.generator.load)
            except Exception as exc:
                logger.exception("Failed to load model")
                self.load_error = f"{type(exc).__name__}: {exc}"

        while True:
            job_id = await self._queue.get()
            job = self._jobs[job_id]
            try:
                await self._run(job)
            finally:
                self._queue.task_done()

    async def _run(self, job: Job) -> None:
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        try:
            if self.load_error:
                raise RuntimeError(f"Model failed to load: {self.load_error}")
            image = Image.open(job.image_path).convert("RGB") if job.image_path else None
            logger.info("Job %s started (%s)", job.id, job.mode)
            await asyncio.to_thread(self.generator.generate, job.params, image, job.video_path)
            job.status = JobStatus.COMPLETED
            logger.info("Job %s completed", job.id)
        except Exception as exc:
            logger.exception("Job %s failed", job.id)
            job.status = JobStatus.FAILED
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            job.finished_at = time.time()
