FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached when only the code changes
COPY requirements_app.txt .
RUN pip install --no-cache-dir -r requirements_app.txt

# App code, serialized model + DictVectorizer, and the script that starts the services
COPY src/web_service/ src/web_service/
COPY bin/run_services.sh bin/run_services.sh

# The Prefect UI runs in the browser of the host, so it must call the API through the host port (4200)
ENV PREFECT_UI_API_URL=http://localhost:4200/api
# Disable the Prefect telemetry service: it writes to the SQLite database at startup and fails with "database is locked"
ENV PREFECT_SERVER_ANALYTICS_ENABLED=false

# 8001: FastAPI app, 4201: Prefect server and UI
EXPOSE 8001 4201

CMD ["bash", "bin/run_services.sh"]
