SHELL := /bin/bash
CWD := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: build-base
build-base:
	@python3 -m venv .venv

.PHONY: build
build: build-base
	@source .venv/bin/activate && \
	pip install -r requirements.txt && \
	pip install -e src/hb-mep

.PHONY: kernel
kernel:
	@source .venv/bin/activate && \
	python -m ipykernel install --user --name=hb-mep-ipython

.PHONY: server
server:
	@source .venv/bin/activate && \
	jupyter notebook .
