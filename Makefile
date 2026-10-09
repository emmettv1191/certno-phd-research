# certno — developer entry points
# Usage: make <target>

PY ?= .venv/bin/python
export PYTHONPATH := src

.PHONY: help install test test-fast lint demo bench1 bench2 bench3 cert al diag clean

help:
	@echo "Targets:"
	@echo "  install    Install package + scientific stack into .venv"
	@echo "  test       Run the full pytest suite"
	@echo "  lint       Byte-compile all sources (syntax check)"
	@echo "  demo       Tiny end-to-end smoke run"
	@echo "  bench1     Elliptic benchmark: solver verification + certificate coverage"
	@echo "  bench2     Burgers benchmark: long-horizon surrogate stability"
	@echo "  bench3     Reaction-diffusion transfer benchmark"
	@echo "  cert       Certificate coverage/sharpness study"
	@echo "  al         Goal-oriented active learning study"
	@echo "  diag       Failure-mode diagnostics (Problem D)"
	@echo "  clean      Remove caches and generated run artifacts"

install:
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"
	$(PY) -m pip install torch --index-url https://download.pytorch.org/whl/cpu

test:
	$(PY) -m pytest

lint:
	$(PY) -m compileall -q src tests experiments

demo:
	$(PY) experiments/run_demo.py

bench1:
	$(PY) experiments/run_benchmark1_elliptic.py --config configs/elliptic.json

bench2:
	$(PY) experiments/run_benchmark2_burgers.py --config configs/burgers.json

bench3:
	$(PY) experiments/run_benchmark3_reaction_diffusion.py --config configs/reaction_diffusion.json

cert:
	$(PY) experiments/run_certificate_study.py --config configs/elliptic.json

al:
	$(PY) experiments/run_active_learning.py --config configs/active_learning.json

diag:
	$(PY) experiments/run_failure_diagnostics.py --config configs/diagnostics.json

clean:
	rm -rf .pytest_cache **/__pycache__ src/**/__pycache__ tests/**/__pycache__
	rm -rf results/runs results/figures/*.png results/data/*.npz
