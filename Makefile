# Simple Makefile for common tasks

.PHONY: help install lint format test coverage docker-build docker-run

help:
	@echo "Targets: install, lint, format, test, coverage, docker-build, docker-run"

install:
	python -m pip install --upgrade pip
	pip install -e .[dev]

lint:
	python -m ruff format --check
	python -m ruff check

format:
	python -m ruff format

test:
	pytest -q

coverage:
	pytest --cov=nvision --cov-report=term-missing --cov-report=xml --cov-report=html

# Docker targets
IMAGE?=nvision:dev

docker-build:
	docker build -t $(IMAGE) -f Dockerfile --target runtime .

docker-run:
	docker run --rm -v %cd%/artifacts:/workspace/artifacts $(IMAGE) --repeats 3 --seed 123 --loc-max-steps 100
