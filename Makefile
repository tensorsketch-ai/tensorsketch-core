.PHONY: install lint format fmt-check type test bench check all

install:            ## Create the environment
	uv sync

lint:               ## Ruff lint
	uv run ruff check src tests examples benchmarks

format:             ## Ruff auto-format
	uv run ruff format src tests examples benchmarks

fmt-check:          ## Ruff format check (no changes)
	uv run ruff format --check src tests examples benchmarks

type:               ## Strict type-check
	uv run mypy

test:               ## Run the test suite
	uv run pytest

bench:              ## Run the micro-benchmarks
	uv run python benchmarks/bench.py

check: lint fmt-check type test  ## Everything CI runs

all: install check
