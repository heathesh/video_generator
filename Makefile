HOST ?= 127.0.0.1
PORT ?= 8000
URL := http://$(HOST):$(PORT)
PROMPT ?= A golden retriever running through a sunlit meadow, slow motion, cinematic lighting
IMAGE ?=
AUDIO ?=
# The audio backend saved by `make setup-audio`, so `make install` keeps its dependencies installed.
AUDIO_BACKEND := $(shell sed -n 's/^VIDEO_AUDIO_BACKEND=//p' .env 2>/dev/null | tr -d '"'"'"' ')
UV_EXTRAS := $(if $(filter mmaudio,$(AUDIO_BACKEND)),--extra mmaudio)

.DEFAULT_GOAL := help
.PHONY: help setup check-system install download-model setup-audio run dev test health example-text example-image example-audio clean

help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: check-system ## First-time setup: check the Mac, install uv if missing, install dependencies
	@command -v uv >/dev/null 2>&1 || { echo "Installing uv with Homebrew..."; brew install uv; }
	@$(MAKE) install

check-system:
	@[ "$$(uname -s)" = "Darwin" ] && [ "$$(uname -m)" = "arm64" ] || { echo "This project targets Apple Silicon Macs (M1 or later)."; exit 1; }
	@command -v brew >/dev/null 2>&1 || command -v uv >/dev/null 2>&1 || { echo "Homebrew not found. Install it from https://brew.sh and re-run 'make setup'."; exit 1; }

install: ## Install Python 3.12 and all dependencies into .venv (uv sync), keeping the chosen audio backend's
	uv sync $(UV_EXTRAS)

download-model: ## Pre-download the model weights (~25 GB) into ~/.cache/huggingface
	uv run python scripts/download_model.py

setup-audio: ## Choose, install and download an audio backend (interactive, or AUDIO=mmaudio|stable-audio|none)
	uv run python scripts/setup_audio.py $(AUDIO)

run: ## Start the API server (override with HOST=... PORT=...)
	uv run uvicorn app.main:app --host $(HOST) --port $(PORT)

dev: ## Start the API server with auto-reload (reloads the model on every code change)
	uv run uvicorn app.main:app --host $(HOST) --port $(PORT) --reload --reload-dir app

test: ## Run the test suite (does not need the model)
	uv run pytest

health: ## Check the running server's status
	@curl -s $(URL)/health; echo

example-text: ## Submit a short text-to-video job (override with PROMPT="...")
	@curl -s -X POST $(URL)/generate -F "prompt=$(PROMPT)" -F num_frames=49; echo

example-image: ## Submit an image-to-video job: make example-image IMAGE=path/to.png PROMPT="..."
	@[ -n "$(IMAGE)" ] || { echo "Usage: make example-image IMAGE=path/to/image.png [PROMPT=\"...\"]"; exit 1; }
	@curl -s -X POST $(URL)/generate -F "prompt=$(PROMPT)" -F "image=@$(IMAGE)" -F num_frames=49; echo

example-audio: ## Submit a text-to-video job with audio (needs make setup-audio first; PROMPT="...")
	@curl -s -X POST $(URL)/generate -F "prompt=$(PROMPT)" -F num_frames=49 -F audio=true; echo

clean: ## Delete generated videos (outputs/) and the virtualenv (.venv)
	rm -rf outputs .venv .pytest_cache
