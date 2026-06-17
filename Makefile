.PHONY: install install-live demo test lint format typecheck run docker-build clean

install:          ## Install core + dev deps (enough for the demo and tests)
	pip install -e ".[dev]"

install-live:     ## Also install the real external-service adapters
	pip install -e ".[live,dev]"

demo:             ## Run the offline, no-API-key retrieval demo
	python -m clipopedia demo

demo-q:           ## Run the demo with a custom query: make demo-q Q="founder burnout"
	python -m clipopedia demo --query "$(Q)"

run:              ## Start the bot loop (needs CLIPOPEDIA_BACKEND=live + .env)
	python -m clipopedia run

test:             ## Run the test suite
	pytest

lint:             ## Lint with ruff
	ruff check .

format:           ## Auto-format with ruff
	ruff format .

typecheck:        ## Static type check with mypy
	mypy src

docker-build:     ## Build the container image
	docker build -t clipopedia:latest .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__ *.egg-info
