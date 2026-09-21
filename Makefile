SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

PY ?= python3.11
VENV := .venv
PIP := $(VENV)/bin/python -m pip
PIP_NO_CACHE := --no-cache-dir
PORT ?= 8000

.PHONY: base env env-cuda12 dev dev-cuda12 clean-docs notebooks docs

base:
	rm -rf $(VENV) build
	@echo "Creating virtual environment with $(PY)..."
	$(PY) -m venv $(VENV)
	@echo "Upgrading pip..."
	$(PIP) install --upgrade pip

env: base
	@echo "Installing package..."
	$(PIP) install $(PIP_NO_CACHE) .

env-cuda12: base
	@echo "Installing package..."
	$(PIP) install $(PIP_NO_CACHE) ".[cuda12]"

dev: base
	@echo "Installing package for development..."
	$(PIP) install $(PIP_NO_CACHE) -e ".[dev]"

dev-cuda12: base
	@echo "Installing package for development..."
	$(PIP) install $(PIP_NO_CACHE) -e ".[dev,cuda12]"

clean-docs:
	@echo "Cleaning docs output..."
	rm -rf docs/build docs/source/api/generated docs/source/tutorials/generated

notebooks:
	@echo "Running tutorial notebooks..."
	find docs/source/tutorials -name '*.ipynb' -print0 | \
	while IFS= read -r -d '' notebook; do \
		$(VENV)/bin/jupyter nbconvert \
			--execute \
			--to notebook \
			--inplace \
			--ExecutePreprocessor.timeout=-1 \
			"$$notebook"; \
	done

docs: clean-docs
	@echo "Building docs with sphinx-autobuild..."
	$(VENV)/bin/sphinx-autobuild --port $(PORT) docs/source docs/build/html
