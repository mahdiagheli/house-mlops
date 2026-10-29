install:
	pip install --upgrade pip && pip install e ".[dev,tracking]"
test:
	python -m pytest -q
lint:
	python -m ruff check .
check: lint test
format:
	black *.py
clean:
	rm -rf .pytest_cache .ruff_cache
all:install lint test format clean