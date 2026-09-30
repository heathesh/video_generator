from pathlib import Path

from scripts.setup_audio import minutes, set_env_value


def test_set_env_value_creates_file(tmp_path: Path):
    env = tmp_path / ".env"
    set_env_value(env, "VIDEO_AUDIO_BACKEND", "mmaudio")
    assert env.read_text() == "VIDEO_AUDIO_BACKEND=mmaudio\n"


def test_set_env_value_replaces_and_keeps_other_lines(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("# my settings\nVIDEO_DTYPE=float32\nVIDEO_AUDIO_BACKEND=mmaudio\nVIDEO_DEFAULT_FPS=12\n")
    set_env_value(env, "VIDEO_AUDIO_BACKEND", "stable-audio")
    assert env.read_text() == (
        "# my settings\nVIDEO_DTYPE=float32\nVIDEO_AUDIO_BACKEND=stable-audio\nVIDEO_DEFAULT_FPS=12\n"
    )


def test_set_env_value_appends_when_missing(tmp_path: Path):
    env = tmp_path / ".env"
    env.write_text("VIDEO_DTYPE=float32")
    set_env_value(env, "VIDEO_AUDIO_BACKEND", "none")
    assert env.read_text() == "VIDEO_DTYPE=float32\nVIDEO_AUDIO_BACKEND=none\n"


def test_minutes():
    assert minutes(10.2, 10) == "~17 min"
    assert minutes(10.2, 3) == "~57 min"
    assert minutes(25, 3) == "~2.3 h"
