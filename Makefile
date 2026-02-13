.PHONY: clean clean-build clean-pyc clean-test help install dev lint format test coverage docs docs-serve release dist

.DEFAULT_GOAL := help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Cleaning ──────────────────────────────────────────────────────

clean: clean-build clean-pyc clean-test ## Remove all build, test, coverage and Python artifacts

clean-build: ## Remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	rm -fr .mesonpy-*
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## Remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## Remove test and coverage artifacts
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

# ── Installation ──────────────────────────────────────────────────

install: ## Install package (editable, with build isolation disabled)
	pip install -e . --no-build-isolation

dev: ## Install package with all dev + docs dependencies
	pip install -e ".[dev,docs]" --no-build-isolation

# ── Quality ───────────────────────────────────────────────────────

lint: ## Check code style with ruff
	ruff check pinball tests

format: ## Auto-format code with ruff
	ruff format pinball tests
	ruff check --fix pinball tests

# ── Testing ───────────────────────────────────────────────────────

test: ## Run tests
	pytest

coverage: ## Run tests with coverage
	pytest --cov=pinball --cov-report=term-missing --cov-report=html
	@echo "Open htmlcov/index.html to view the report"

# ── Documentation ─────────────────────────────────────────────────

docs: ## Build documentation with MkDocs
	mkdocs build

docs-serve: ## Serve documentation locally with live reload
	mkdocs serve

# ── Distribution ──────────────────────────────────────────────────

dist: clean ## Build source and wheel distributions
	python -m build
	ls -l dist

release: dist ## Upload to PyPI (requires twine)
	twine upload dist/*
