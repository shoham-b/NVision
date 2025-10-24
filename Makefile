# Simple Makefile for common tasks

.PHONY: help install lint format test coverage docker-build docker-run

help:
	@echo "Targets: install, lint, format, test, coverage, docker-build, docker-run"

install:
	uv sync --group dev

lint:
	uv run ruff format --check
	uv run ruff check

format:
	uv run ruff format

test:
	uv run pytest -q

coverage:
	uv run pytest --cov=nvision --cov-report=term-missing --cov-report=xml --cov-report=html

# Docker targets
IMAGE?=nvision:dev

docker-build:
	docker build -t $(IMAGE) -f Dockerfile --target runtime .

docker-run:
	docker run --rm -v %cd%/artifacts:/workspace/artifacts $(IMAGE) --repeats 3 --seed 123 --loc-max-steps 100
