SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

PY ?= python3.11
VENV := .venv
PIP := $(VENV)/bin/python -m pip

.PHONY: base env dev docs

base:
	rm -rf $(VENV) build
	@echo "Creating virtual environment with $(PY)..."
	$(PY) -m venv $(VENV)
	@echo "Upgrading pip..."
	$(PIP) install --upgrade pip

env: base
	@echo "Installing package..."
	$(PIP) install .

dev: base
	@echo "Installing package for development..."
	$(PIP) install -e ".[dev]"

docs:
	@echo "Building docs with sphinx-autobuild..."
	$(VENV)/bin/sphinx-autobuild docs/source docs/build/html
