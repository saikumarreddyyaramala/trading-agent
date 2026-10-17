.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-ui test-integration test-paper-orders check healthcheck docker-build docker-healthcheck

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## Install dependencies and git hooks
	uv sync
	uv run pre-commit install

lint: ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-format and fix lint issues
	uv run ruff format .
	uv run ruff check --fix .

typecheck: ## Static type checking (mypy --strict)
	uv run mypy

test: ## Unit tests with coverage (no network, no keys)
	uv run pytest --cov --cov-report=term-missing --cov-fail-under=90

test-integration: ## Tests against real Alpaca paper + LLM (needs .env)
	uv run pytest -m integration -v

test-paper-orders: ## Round-trip a real order on the PAPER account (market hours)
	RUN_PAPER_ORDERS=1 uv run pytest -m paper_orders -v -s

test-ui: ## Dashboard UI logic tests (Node's built-in runner, no npm packages)
	node --test tests/js/*.test.mjs

check: lint typecheck test test-ui ## Everything CI runs

healthcheck: ## Pre-flight check using your .env
	uv run trading-agent healthcheck

docker-build: ## Build the production image
	docker build -t trading-agent:local .

docker-healthcheck: docker-build ## Run the health check inside the container
	docker run --rm --env-file .env trading-agent:local healthcheck
