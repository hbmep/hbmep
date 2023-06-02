SHELL := /bin/bash
CWD := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

python ?= 3.9
model ?= baseline

export

.PHONY: check-env
check-env:
PYTHON3_OK := $(shell python3 --version 2>&1)
ifeq ('$(PYTHON3_OK)','')
    $(error package 'python3' not found)
endif

.PHONY: build-base
build-base: check-env
	@python$(python) -m venv .venv

.PHONY: build
build: build-base
	@source .venv/bin/activate && \
	pip install --upgrade pip && \
	pip install -r requirements.txt && \
	pip install -e src/hb-mep

run:
	@source .venv/bin/activate && \
	python -m hb_mep run --model=$(model)

.PHONY: kernel
kernel:
	@source .venv/bin/activate && \
	python -m ipykernel install --user --name=hb-mep-ipython

.PHONY: server
server:
	@source .venv/bin/activate && \
	jupyter notebook .
