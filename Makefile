.PHONY: install lint format format-check typecheck test check fix audit clean

install:
	uv sync
	uv run pre-commit install

load-docs:
	uv run python -m scripts.doc_loader

lint:
	uv run ruff check

format:
	uv run ruff format

format-check:
	uv run ruff format --check

typecheck:
	uv run ty check

test:
	uv run pytest

check: lint format-check typecheck test

fix:
	uv run ruff check --fix
	uv run ruff format

audit:
	uv audit

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .ruff_cache .pytest_cache
