
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
projDir:
	mkdir -p src/houseprice_mlops/pipelines
	mkdir -p src/houseprice_mlops/evaluation
	mkdir -p src/houseprice_mlops/monitoring
	mkdir -p tests
	mkdir -p .github/workflows
	touch src/houseprice_mlops/__init__.py
	touch src/houseprice_mlops/pipelines/__init__.py
	touch src/houseprice_mlops/pipelines/train.py
	touch src/houseprice_mlops/evaluation/__init__.py
	touch src/houseprice_mlops/evaluation/quality_gate.py
	touch src/houseprice_mlops/monitoring/__init__.py
	touch src/houseprice_mlops/monitoring/metrics.py
	touch src/houseprice_mlops/api.py
	touch .github/workflows/pipeline.yml
