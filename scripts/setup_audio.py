"""Choose, install and download an audio backend, then save the choice to .env as VIDEO_AUDIO_BACKEND.

Usage:
  uv run python scripts/setup_audio.py                      # interactive: shows the options and asks
  uv run python scripts/setup_audio.py mmaudio              # pick directly (still asks you to accept the licence)
  uv run python scripts/setup_audio.py mmaudio --accept-license
"""

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"
ENV_KEY = "VIDEO_AUDIO_BACKEND"
HUNYUAN_DIR = ROOT / "backends" / "hunyuan_foley"
HUNYUAN_REPO = "https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley"
HUNYUAN_COMMIT = "df7b005b5023df2a9b73e1d66dd51d452799884e"


@dataclass(frozen=True)
class Option:
    name: str
    model: str
    synced: str
    commercial: str
    size_gb: float
    licence: str
    licence_url: str
    extra: str | None = None  # uv extra holding its dependencies
    separate_env: bool = False  # installed into its own environment under backends/ instead
    hf_repo: str | None = None  # gated Hugging Face repo that needs its terms accepted first


OPTIONS = [
    Option(
        name="mmaudio",
        model="MMAudio large_44k_v2",
        synced="yes",
        commercial="no",
        size_gb=10.7,
        licence="CC-BY-NC 4.0: free for non-commercial use only. You can't use the audio commercially.",
        licence_url="https://huggingface.co/hkchengrex/MMAudio",
        extra="mmaudio",
    ),
    Option(
        name="hunyuan-foley",
        model="HunyuanVideo-Foley XL",
        synced="yes",
        commercial="yes, with limits",
        size_gb=10.6,
        licence=(
            "Tencent Hunyuan Community License: commercial use is allowed, except that the licence does not apply in\n"
            "the European Union, the United Kingdom or South Korea, and products with over 100 million monthly users\n"
            "need a separate licence from Tencent. Output you share must be clearly marked as AI-generated, and the\n"
            "acceptable use policy applies."
        ),
        licence_url="https://huggingface.co/tencent/HunyuanVideo-Foley/blob/main/LICENSE",
        separate_env=True,
    ),
    Option(
        name="stable-audio",
        model="Stable Audio Open 1.0",
        synced="no (prompt only)",
        commercial="under US$1M revenue",
        size_gb=5.3,
        licence=(
            "Stability AI Community License: commercial use is free if you (with affiliates) earn under US$1M a year,\n"
            "but you must register with Stability AI and show 'Powered by Stability AI' if you distribute it.\n"
            "Above US$1M you need an enterprise licence (https://stability.ai/enterprise)."
        ),
        licence_url="https://huggingface.co/stabilityai/stable-audio-open-1.0/blob/main/LICENSE.md",
        hf_repo="stabilityai/stable-audio-open-1.0",
    ),
]
BY_NAME = {o.name: o for o in OPTIONS}
FUTURE = [
    "LTX-2 (video and synced audio in one model; free commercial use under US$10M revenue) needs 64 GB+ of memory.",
]


def set_env_value(path: Path, key: str, value: str) -> None:
    """Set key=value in a .env file, replacing an existing entry and leaving every other line alone."""
    lines = path.read_text().splitlines() if path.exists() else []
    entry = f"{key}={value}"
    for i, line in enumerate(lines):
        if line.split("=", 1)[0].strip() == key:
            lines[i] = entry
            break
    else:
        lines.append(entry)
    path.write_text("\n".join(lines) + "\n")


def minutes(size_gb: float, mb_per_s: float) -> str:
    total = round(size_gb * 1000 / mb_per_s / 60)
    return f"~{total} min" if total < 60 else f"~{total / 60:.1f} h"


def print_options() -> None:
    print("Audio options (the audio is added to a finished video when a request sets audio=true):\n")
    header = f"  {'option':<15}{'model':<24}{'synced':<19}{'commercial use':<22}{'download':<10}{'@10 MB/s':<10}@3 MB/s"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for o in OPTIONS:
        print(f"  {o.name:<15}{o.model:<24}{o.synced:<19}{o.commercial:<22}{f'{o.size_gb:.1f} GB':<10}"
              f"{minutes(o.size_gb, 10):<10}{minutes(o.size_gb, 3)}")
    print(f"  {'none':<15}turn audio off")
    print("\nNot available here yet:")
    for note in FUTURE:
        print(f"  - {note}")
    print()


def choose() -> str:
    print_options()
    names = [o.name for o in OPTIONS] + ["none"]
    while True:
        answer = input(f"Choose an option ({' / '.join(names)}): ").strip().lower()
        if answer in names:
            return answer
        print(f"Please type one of: {', '.join(names)}")


def accept_licence(option: Option) -> bool:
    print(f"\n{option.model} licence\n  {option.licence.replace(chr(10), chr(10) + '  ')}\n  Full terms: {option.licence_url}\n")
    return input("Do you accept these terms? [y/N]: ").strip().lower() in {"y", "yes"}


def check_hf_access(repo: str) -> bool:
    from huggingface_hub import auth_check
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    try:
        auth_check(repo)
        return True
    except (GatedRepoError, RepositoryNotFoundError):
        print(
            f"\nYour Hugging Face account can't download {repo} yet. It's gated:\n"
            f"  1. Open https://huggingface.co/{repo}, sign in and accept the terms.\n"
            f"  2. Create a Read token at https://huggingface.co/settings/tokens and run: uv run hf auth login\n"
            f"  3. Run make setup-audio again."
        )
        return False


def run(*cmd: str, cwd: Path = ROOT) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def install_hunyuan_foley() -> None:
    """Check out the pinned upstream source, patch it for Apple Silicon and build its separate environment."""
    src = HUNYUAN_DIR / "src"
    if not src.exists():
        run("git", "clone", "--quiet", HUNYUAN_REPO, str(src))
    run("git", "fetch", "--quiet", "origin", HUNYUAN_COMMIT, cwd=src)
    run("git", "checkout", "--quiet", "--force", HUNYUAN_COMMIT, cwd=src)  # --force also resets the old patch
    run("git", "apply", str(HUNYUAN_DIR / "mps.patch"), cwd=src)
    run("uv", "sync", "--frozen", cwd=HUNYUAN_DIR)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    accepted = "--accept-license" in sys.argv
    choice = args[0] if args else choose()

    if choice == "none":
        run("uv", "sync")  # drops any audio extra
        set_env_value(ENV_FILE, ENV_KEY, "none")
        print(f"\nAudio is off ({ENV_KEY}=none in .env). Restart the server to apply it.")
        return 0
    if choice not in BY_NAME:
        print(f"Unknown option {choice!r}. Choose from: {', '.join([*BY_NAME, 'none'])}")
        return 2

    option = BY_NAME[choice]
    if not accepted and not accept_licence(option):
        print("Not accepted; nothing was changed.")
        return 1
    if option.hf_repo and not check_hf_access(option.hf_repo):
        return 1

    print(f"\nInstalling dependencies for {option.name} ...")
    run("uv", "sync", *(["--extra", option.extra] if option.extra else []))
    if option.separate_env:
        install_hunyuan_foley()
    print(f"\nDownloading the {option.model} weights (~{option.size_gb:.1f} GB, one time only) ...")
    run("uv", "run", "python", "scripts/download_model.py", f"--audio={option.name}")

    set_env_value(ENV_FILE, ENV_KEY, option.name)
    print(f"\nDone: {ENV_KEY}={option.name} is saved in .env. Restart the server (make run), then add audio=true to a request.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
