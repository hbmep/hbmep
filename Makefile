SHELL := /bin/bash
CWD := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

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

.PHONY: venv-test
venv-test: venv-base
	@source .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -e .[test]

.PHONY: venv-docs
venv-docs: venv-base
	@source .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -e .[docs]

.PHONY: clean
clean:
	rm -rf .venv

run:
	@source .venv/bin/activate && \
	python -m hbmep $(config)

html:
	@source .venv/bin/activate && \
	sphinx-build -M html docs/source docs/build

test:
	@source .venv/bin/activate && \
	pytest
