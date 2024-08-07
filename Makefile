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

.PHONY: clean
clean:
	rm -rf .venv

run:
	@source .venv/bin/activate && \
	python -m hbmep $(config)

html:
	@source .venv/bin/activate && \
	sphinx-autobuild docs/source/ docs/build/html/

test:
	@source .venv/bin/activate && \
	pytest
