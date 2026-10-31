
.PHONY: install test lint check format clean all

install:
	pip install --upgrade pip
	pip install -e ".[dev,tracking,api]"

test:
	python -m pytest -q

lint:
	python -m ruff check .

check: lint test

format:
	python -m ruff format .

clean:
	rm -rf .pytest_cache .ruff_cache

all: install check format clean
