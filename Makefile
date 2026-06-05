# Crucible — developer task runner.
#
# Quick start (clean machine):
#     make setup        # create .venv and install dev dependencies
#     make test         # run the test suite
#     make smoke        # run one agent task end-to-end (offline, mock LLM)
#     make bench-smoke  # run the benchmark pipeline shake-out (offline)
#
# `make setup` creates a local .venv. The other targets prefer that venv's
# Python if it exists, and otherwise fall back to whatever `python3` is active —
# so they also work inside an already-activated environment.

VENV   ?= .venv
PYTHON ?= python3
# Immediate (:=) on purpose: resolve the interpreter once at parse time so a
# `make test` immediately after `make setup` picks up the freshly created venv.
PY     := $(shell [ -x "$(VENV)/bin/python" ] && echo "$(VENV)/bin/python" || echo $(PYTHON))

.DEFAULT_GOAL := help
.PHONY: help setup test cov smoke bench-smoke clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## One-command setup: create .venv and install dev dependencies
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install -r requirements-dev.txt
	@echo "✅ Setup complete. Try: make test"

test: ## Run the test suite (quiet)
	$(PY) -m pytest -q

cov: ## Run the test suite with coverage
	$(PY) -m pytest --cov=agent

smoke: ## Run one agent task end-to-end with the offline mock LLM
	$(PY) -m agent "add two numbers" --llm mock

bench-smoke: ## Run the benchmark pipeline offline (mock LLM, no Docker)
	$(PY) -m bench.runner --smoke --llm mock --no-docker --quiet

clean: ## Remove caches and generated artifacts
	rm -rf .pytest_cache .agent_state .agent_memory
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
