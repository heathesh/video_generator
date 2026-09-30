import io
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.generator import VideoParams, snap_dimension, snap_num_frames
from app.main import create_app

FAKE_VIDEO = b"\x00\x00\x00\x18ftypmp42fake-video"


class FakeGenerator:
    """Stands in for the real model: records calls and writes placeholder bytes instead of a video."""

    model_id = "fake/model"
    device = "cpu"

    def __init__(self, fail: bool = False):
        self.loaded = False
        self.fail = fail
        self.calls: list[tuple[VideoParams, Image.Image | None]] = []
        self.release = threading.Event()
        self.release.set()

    def load(self) -> None:
        self.loaded = True

    def generate(self, params: VideoParams, image: Image.Image | None, output_path: Path) -> Path:
        self.release.wait(timeout=5)
        self.calls.append((params, image))
        if self.fail:
            raise RuntimeError("boom")
        output_path.write_bytes(FAKE_VIDEO)
        return output_path


@pytest.fixture
def generator() -> FakeGenerator:
    return FakeGenerator()


@pytest.fixture
def client(generator: FakeGenerator, tmp_path: Path):
    # audio_backend is explicit so a developer's .env can't change the results.
    app = create_app(Settings(output_dir=tmp_path, audio_backend="none"), generator)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def audio_client(generator: FakeGenerator, tmp_path: Path, request: pytest.FixtureRequest):
    """A client whose server has an audio backend configured (mmaudio unless parametrized indirectly)."""
    backend = getattr(request, "param", "mmaudio")
    app = create_app(Settings(output_dir=tmp_path, audio_backend=backend), generator)
    with TestClient(app) as c:
        yield c


def wait_for(client: TestClient, job_id: str, status: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    job: dict = {}
    while time.monotonic() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] == status:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached {status!r}; last state: {job}")


def png_bytes(size=(100, 60), fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "red").save(buf, format=fmt)
    return buf.getvalue()


def test_health_reports_model(client: TestClient):
    wait = time.monotonic() + 5
    while not client.get("/health").json()["model_loaded"] and time.monotonic() < wait:
        time.sleep(0.02)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_id"] == "fake/model"
    assert body["device"] == "cpu"


def test_text_to_video_lifecycle(client: TestClient, generator: FakeGenerator):
    resp = client.post("/generate", data={"prompt": "a cat on a beach", "seed": 7})
    assert resp.status_code == 202
    job = resp.json()
    assert job["mode"] == "text-to-video"
    assert job["status"] in {"queued", "running", "completed"}

    job = wait_for(client, job["job_id"], "completed")
    assert job["video_url"] == f"/jobs/{job['job_id']}/video"

    video = client.get(job["video_url"])
    assert video.status_code == 200
    assert video.headers["content-type"] == "video/mp4"
    assert video.content == FAKE_VIDEO

    params, image = generator.calls[0]
    assert image is None
    assert params.prompt == "a cat on a beach"
    assert params.seed == 7


def test_image_to_video_passes_image(client: TestClient, generator: FakeGenerator):
    resp = client.post(
        "/generate",
        data={"prompt": "the flower sways in the wind"},
        files={"image": ("flower.jpg", png_bytes(fmt="JPEG"), "image/jpeg")},
    )
    assert resp.status_code == 202
    assert resp.json()["mode"] == "image-to-video"

    wait_for(client, resp.json()["job_id"], "completed")
    _, image = generator.calls[0]
    assert image is not None
    assert image.size == (100, 60)


def test_params_are_snapped_to_model_constraints(client: TestClient):
    resp = client.post("/generate", data={"prompt": "x", "width": 700, "height": 500, "num_frames": 50})
    params = resp.json()["params"]
    assert (params["width"], params["height"], params["num_frames"]) == (672, 480, 49)


def test_defaults_applied(client: TestClient):
    params = client.post("/generate", data={"prompt": "x"}).json()["params"]
    defaults = Settings()
    assert params["width"] == defaults.default_width
    assert params["num_frames"] == defaults.default_num_frames
    assert params["negative_prompt"] == defaults.default_negative_prompt


def test_audio_off_by_default(client: TestClient):
    params = client.post("/generate", data={"prompt": "x"}).json()["params"]
    assert params["audio"] is False
    assert params["audio_prompt"] is None


def test_audio_rejected_when_not_set_up(client: TestClient):
    resp = client.post("/generate", data={"prompt": "x", "audio": "true"})
    assert resp.status_code == 422
    assert "setup-audio" in resp.json()["detail"]
    assert client.get("/health").json()["audio_backend"] == "none"


def test_audio_params_passed(audio_client: TestClient, generator: FakeGenerator):
    resp = audio_client.post(
        "/generate",
        data={
            "prompt": "a glass shatters",
            "audio": "true",
            "audio_prompt": " glass breaking ",
            "audio_num_inference_steps": 10,
        },
    )
    assert resp.status_code == 202
    assert resp.json()["params"]["audio"] is True

    wait_for(audio_client, resp.json()["job_id"], "completed")
    params, _ = generator.calls[0]
    assert params.audio is True
    assert params.audio_prompt == "glass breaking"
    assert params.audio_num_inference_steps == 10


@pytest.mark.parametrize(
    ("audio_client", "steps"),
    [("mmaudio", 25), ("stable-audio", 100), ("hunyuan-foley", 50)],
    indirect=["audio_client"],
)
def test_audio_defaults_come_from_backend(audio_client: TestClient, steps: int):
    params = audio_client.post("/generate", data={"prompt": "x", "audio": "true", "audio_prompt": "  "}).json()["params"]
    assert params["audio_prompt"] is None
    assert params["audio_num_inference_steps"] == steps


@pytest.mark.parametrize(
    ("audio_client", "status_code"),
    [("mmaudio", 422), ("stable-audio", 202), ("hunyuan-foley", 202)],
    indirect=["audio_client"],
)
def test_audio_length_limit_depends_on_backend(audio_client: TestClient, status_code: int):
    # 257 frames at 24 fps is ~10.7 s: over MMAudio's 10 s limit, within Hunyuan's 15 s and Stable Audio's 47 s.
    data = {"prompt": "x", "num_frames": 257}
    assert audio_client.post("/generate", data={**data, "audio": "true"}).status_code == status_code
    assert audio_client.post("/generate", data=data).status_code == 202


def test_invalid_image_rejected(client: TestClient):
    resp = client.post(
        "/generate",
        data={"prompt": "x"},
        files={"image": ("notes.txt", b"definitely not an image", "text/plain")},
    )
    assert resp.status_code == 400


@pytest.mark.parametrize(
    "data",
    [{}, {"prompt": "   "}, {"prompt": "x", "num_frames": 1}, {"prompt": "x", "width": 99999}],
)
def test_invalid_requests_rejected(client: TestClient, data: dict):
    assert client.post("/generate", data=data).status_code == 422


def test_video_not_ready_returns_409(client: TestClient, generator: FakeGenerator):
    generator.release.clear()
    job_id = client.post("/generate", data={"prompt": "x"}).json()["job_id"]
    assert client.get(f"/jobs/{job_id}/video").status_code == 409
    generator.release.set()
    wait_for(client, job_id, "completed")


def test_unknown_job_returns_404(client: TestClient):
    assert client.get("/jobs/nope").status_code == 404
    assert client.get("/jobs/nope/video").status_code == 404


def test_failed_generation_reports_error(tmp_path: Path):
    app = create_app(Settings(output_dir=tmp_path), FakeGenerator(fail=True))
    with TestClient(app) as client:
        job_id = client.post("/generate", data={"prompt": "x"}).json()["job_id"]
        job = wait_for(client, job_id, "failed")
        assert "boom" in job["error"]
        assert client.get(f"/jobs/{job_id}/video").status_code == 409


@pytest.mark.parametrize(("value", "expected"), [(704, 704), (500, 480), (10, 32)])
def test_snap_dimension(value: int, expected: int):
    assert snap_dimension(value) == expected


@pytest.mark.parametrize(("value", "expected"), [(97, 97), (100, 97), (9, 9), (1, 9), (16, 9), (17, 17)])
def test_snap_num_frames(value: int, expected: int):
    assert snap_num_frames(value) == expected
