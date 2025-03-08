SHELL := /bin/bash
CWD := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
PY_VERSION ?= 3.11

define activate_venv
	@echo "Activating virtual environment and installing dependencies..."
	source .venv/bin/activate && \
	pip install --upgrade pip
endef

.PHONY: base
base:
	rm -rf .venv build
	@echo "Creating virtual environment..."
	python$(PY_VERSION) -m venv .venv

.PHONY: build
build: base
	$(activate_venv) && \
	pip install .

.PHONY: dev
dev: base
	$(activate_venv) && \
	pip install -e .[dev]

.PHONY: rsch
rsch: base
	$(activate_venv) && \
	pip install -e .[rsch]
