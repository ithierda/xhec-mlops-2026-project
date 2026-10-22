.PHONY: install lint mlflow-ui prefect-ui train deploy api docker-build docker-run

export PREFECT_API_URL = http://127.0.0.1:4200/api

IMAGE_NAME = abalone-api:0.0.1

install:
	uv sync
	uv run pre-commit install

lint:
	uv run pre-commit run --all-files

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
