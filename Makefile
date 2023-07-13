SHELL := /bin/bash
CWD := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

export

PY_VERSION ?= 3.11


.PHONY: check-env
check-env:
PYTHON3_OK := $(shell python3 --version 2>&1)
ifeq ('$(PYTHON3_OK)','')
    $(error package 'python3' not found)
endif

.PHONY: venv-base
venv-base: check-env
	@python$(PY_VERSION) -m venv .venv

.PHONY: venv
venv: venv-base
	@source .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install .

.PHONY: venv-dev
venv-dev: venv-base
	@source .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -e .[dev]

.PHONY: venv-docs
venv-docs: venv-base
	@source .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -e .[docs]

.PHONY: clean
clean:
	rm -rf .venv
