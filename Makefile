.DEFAULT_GOAL := help

.PHONY: help install format lint typecheck test test-cov ci ci-strict docs docs-serve

COV_MODULE = pydantic_team
RUFF_PATHS = pydantic_team tests

help: ## Show available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Sync package, lint, and dev dependencies; install pre-commit
	uv sync --group lint --group dev
	@uv run pre-commit install 2>/dev/null || true

format: ## Format code with ruff
	uv run ruff format $(RUFF_PATHS)
	uv run ruff check --fix --fix-only $(RUFF_PATHS)

lint: ## Check formatting and lint (no write)
	uv run ruff format --check $(RUFF_PATHS)
	uv run ruff check $(RUFF_PATHS)

typecheck: ## Run pyright strict type checking
	uv run pyright

test: ## Unit tests with 100% coverage gate
	uv run pytest -q --cov=$(COV_MODULE) --cov-report=term-missing --cov-fail-under=100

test-cov: test ## Alias for test (coverage required)

ci: ## Pre-PR gate: format, lint, tests with coverage
	@set -e; \
	$(MAKE) format; \
	$(MAKE) lint; \
	$(MAKE) test

ci-strict: ## CI plus typecheck
	@set -e; \
	$(MAKE) lint; \
	$(MAKE) typecheck; \
	$(MAKE) test

docs: ## Build documentation site into site/
	uv sync --group docs
	uv run mkdocs build --strict

docs-serve: ## Serve documentation locally (live reload)
	uv sync --group docs
	uv run mkdocs serve
