SHELL := /bin/bash
CWD := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

export

python ?= 3.11
job ?= inference
model ?= Baseline
dataset ?= rats
tag ?= latest

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
	pip install -e .

run:
	@source .venv/bin/activate && \
	python -m hbmep run \
	--job=$(job) \
	--model=$(model) \
	--dataset=$(dataset) \
	--id=$(id) \
	--tag=$(tag)
