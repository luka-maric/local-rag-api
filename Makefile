.PHONY: install setup test test-cov dev infra down reset logs shell clean download-models pull-llm import-llm prometheus grafana ui build migrate up up-corporate status

setup:
	cp .env.example .env
	@python3 -c "import secrets; c=open('.env').read(); open('.env','w').write(c.replace('change-this-in-production-use-openssl-rand-hex-32', secrets.token_hex(32)))"
	@echo ".env created. JWT_SECRET_KEY generated."

install:
	pip install --upgrade setuptools pip
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=app --cov-report=term-missing

# Full dev stack with hot reload and debugpy
dev:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up

# Infrastructure only (Postgres + Redis) — for running the app locally outside Docker
infra:
	docker compose up postgres redis -d

down:
	docker compose down

# Full reset — stops all containers and deletes data volumes
reset:
	docker compose down -v

logs:
	docker compose logs -f

shell:
	docker compose exec api bash

# Pre-download both ML models locally (optional — avoids stall on first upload)
download-models:
	python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('Embedding model cached.')"
	python -c "from transformers import pipeline; pipeline('ner', model='dslim/bert-base-NER', aggregation_strategy='simple'); print('NER model cached.')"

# Pull the LLM into the Ollama container — requires direct internet access
pull-llm:
	@echo "Waiting for Ollama to be ready..."
	@until docker compose exec ollama ollama list > /dev/null 2>&1; do sleep 3; done
	docker compose exec ollama ollama pull llama3.2:3b
	@$(MAKE) status

# Import a local GGUF as llama3.2:3b — use on corporate / air-gapped networks
# Usage: make import-llm GGUF=~/path/to/model.gguf
import-llm:
	@test -n "$(GGUF)" || (echo "Usage: make import-llm GGUF=/path/to/file.gguf" && exit 1)
	@echo "Waiting for Ollama to be ready..."
	@until docker compose exec ollama ollama list > /dev/null 2>&1; do sleep 3; done
	docker compose cp ollama/Modelfile ollama:/tmp/Modelfile
	docker compose exec -T ollama sh -c "cp /dev/stdin /tmp/model.gguf" < $(GGUF)
	docker compose exec ollama ollama create llama3.2:3b -f /tmp/Modelfile
	@$(MAKE) status

prometheus:
	xdg-open http://localhost:9090

# Credentials: admin / admin
grafana:
	xdg-open http://localhost:3000

ui:
	xdg-open http://localhost:7860

up:
	docker compose up -d

# Corporate path — mounts system CA bundle for TLS-intercepting proxies.
# Set SYSTEM_CA_BUNDLE in .env before running. Never use plain 'make up' on corporate machines.
up-corporate:
	docker compose -f docker-compose.yml -f docker-compose.corporate.yml up -d

# Rebuild images (api + ui). Follow with 'make up' / 'make up-corporate' to deploy.
# Note: 'docker compose restart' does NOT pick up rebuilt images — 'up -d' does.
build:
	docker compose build api ui

# Run Alembic migrations inside the running API container
migrate:
	docker compose exec api alembic upgrade head

status:
	@echo ""
	@echo "=================================================="
	@echo "  RAG API — Stack Status"
	@echo "=================================================="
	@docker compose ps --format "  {{.Service}}\t{{.Status}}" 2>/dev/null
	@echo ""
	@echo "  Services:"
	@echo "    Gradio UI    http://localhost:7860"
	@echo "    API / Docs   http://localhost:8000/docs"
	@echo "    Grafana      http://localhost:3000  (admin / admin)"
	@echo "    Prometheus   http://localhost:9090"
	@echo ""
	@curl -sf http://localhost:8000/health > /dev/null \
		&& echo "  API health:  OK — stack is ready" \
		|| echo "  API health:  not yet ready (run 'make logs' to check)"
	@echo ""

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
