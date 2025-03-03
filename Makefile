SHELL := /bin/bash
CWD := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

PY_VERSION ?= 3.11
PORT ?= 8000

.PHONY: check-env
check-env:
PYTHON3_OK := $(shell python3 --version 2>&1)
ifeq ('$(PYTHON3_OK)','')
    $(error package 'python3' not found)
endif

.PHONY: venv-base
venv-base: check-env
	@rm -rf .venv
	@python$(PY_VERSION) -m venv .venv

.PHONY: build
build: venv-base
	@source .venv/bin/activate && \
	pip install --upgrade pip && \
	if [ -z "$(ENV)" ]; then \
		pip install .; \
	else \
		pip install -e ".[$(ENV)]"; \
	fi

.PHONY: clean
clean:
	rm -rf .venv

run:
	@source .venv/bin/activate && \
	python -m hbmep $(config)

html:
	@source .venv/bin/activate && \
	sphinx-autobuild docs/source/ docs/build/html/ --port $(PORT)

test:
	@source .venv/bin/activate && \
	pytest
