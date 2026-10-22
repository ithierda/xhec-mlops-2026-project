.PHONY: all install lint test mlflow-ui prefect-ui train deploy api docker-build docker-run

export PREFECT_API_URL = http://127.0.0.1:4200/api

IMAGE_NAME = abalone-api:0.0.1

# Install, check, test, train, then build and run the Docker image (API + Prefect UI)
# The training runs with an empty PREFECT_API_URL, so Prefect uses a temporary local server and `make prefect-ui` is not needed
all: install lint test
	$(MAKE) train PREFECT_API_URL=
	$(MAKE) docker-build
	$(MAKE) docker-run

install:
	uv sync
	uv run pre-commit install

lint:
	uv run pre-commit run --all-files

test:
	uv run pytest

mlflow-ui:
	uv run mlflow ui --backend-store-uri mlruns

prefect-ui:
	uv run prefect server start --host 127.0.0.1 --port 4200

train:
	uv run python src/modelling/main.py data/abalone.csv

deploy:
	uv run python src/modelling/deployment.py

api:
	uv run uvicorn src.web_service.main:app --reload --port 8000

docker-build:
	docker build -t $(IMAGE_NAME) -f Dockerfile.app .

docker-run:
	docker run --rm -p 0.0.0.0:8000:8001 -p 0.0.0.0:4200:4201 $(IMAGE_NAME)
