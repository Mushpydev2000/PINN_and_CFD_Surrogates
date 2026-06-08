# Makefile for PINN and CFD Surrogate Project

.PHONY: help install install-dev install-notebooks install-tracking lint format test train-pinn train-surrogate train-both clean clean-all docs notebook

help:
	@echo "Physics-Informed Neural Networks and CFD Surrogate Modeling"
	@echo ""
	@echo "Available commands:"
	@echo "  make install              - Install package dependencies"
	@echo "  make install-dev          - Install package with dev tools"
	@echo "  make install-notebooks    - Install notebook dependencies"
	@echo "  make install-all          - Install everything"
	@echo "  make lint                 - Run pylint and mypy"
	@echo "  make format               - Format code with black"
	@echo "  make test                 - Run unit tests"
	@echo "  make train-pinn           - Train PINN model"
	@echo "  make train-surrogate      - Train surrogate model"
	@echo "  make train-both           - Train both models"
	@echo "  make clean                - Clean cache/logs/checkpoints"
	@echo "  make clean-all            - Clean everything including data"
	@echo "  make docs                 - Build documentation"
	@echo ""

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"

install-notebooks:
	pip install -e ".[notebooks]"

install-tracking:
	pip install -e ".[tracking]"

install-all:
	pip install -e ".[dev,notebooks,tracking]"

# Code Quality
lint:
	@echo "Running pylint..."
	pylint src/ --disable=C0111,C0103,R0913,R0914 || true
	@echo "Running mypy..."
	mypy src/ || true

format:
	@echo "Formatting code with black..."
	black src/ tests/ --line-length=100

format-check:
	black src/ tests/ --check --line-length=100

# Testing
test:
	pytest tests/ -v --cov=src --cov-report=html

test-quick:
	pytest tests/ -v -x

# Training
train-pinn:
	@echo "Training PINN model..."
	python src/main.py --mode pinn --pinn-config configs/pinn_config.yaml

train-surrogate:
	@echo "Training surrogate model..."
	python src/main.py --mode surrogate --surrogate-config configs/surrogate_config.yaml

train-both:
	@echo "Training both PINN and surrogate models..."
	python src/main.py --mode both --pinn-config configs/pinn_config.yaml --surrogate-config configs/surrogate_config.yaml

# Cleanup
clean:
	@echo "Cleaning cache, logs, and checkpoints..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov
	rm -rf logs/*.log
	rm -rf checkpoints/*.pt

clean-data:
	@echo "Cleaning generated data..."
	rm -rf data/generated/*
	rm -rf data/processed/*

clean-all: clean clean-data
	@echo "Cleaning everything..."
	rm -rf build/ dist/ *.egg-info/
	rm -rf figures/*.png figures/*.pdf
	rm -rf reports/*.pdf

# Documentation
docs:
	@echo "Building documentation..."
	@echo "Main documentation: README.md"
	@echo "Report template: reports/project_report_template.md"
	@echo "Experiment log: reports/experiment_log.md"

# Jupyter Notebooks
notebook:
	jupyter notebook notebooks/

notebook-lab:
	jupyter lab notebooks/

# Development utilities
setup-git:
	@echo "Setting up git hooks..."
	pre-commit install || true

check: format-check lint test
	@echo "All checks passed!"

# Utility commands
list-checkpoints:
	@echo "Available checkpoints:"
	@ls -lh checkpoints/*.pt 2>/dev/null || echo "No checkpoints found"

list-logs:
	@echo "Recent logs:"
	@ls -lh logs/*.log 2>/dev/null | tail -10 || echo "No logs found"

tensorboard:
	tensorboard --logdir logs/

# GPU utilities
cuda-info:
	python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Python environment
requirements:
	pip freeze > requirements-freeze.txt
	@echo "Frozen requirements saved to requirements-freeze.txt"

venv:
	python3 -m venv venv
	@echo "Virtual environment created. Activate with: source venv/bin/activate"

# Docker (optional future extension)
docker-build:
	docker build -t pinn-surrogate:latest .

docker-run:
	docker run -it --gpus all pinn-surrogate:latest

# Performance profiling
profile:
	@echo "Run python -m cProfile -s cumtime src/main.py to profile"

benchmark:
	python -c "from src.pinn import PINNNetwork; import torch; model = PINNNetwork(); x = torch.randn(1000, 2); import time; t = time.time(); _ = model(x); print(f'1000 samples: {(time.time()-t)*1000:.2f}ms')"
