#!/bin/bash

# Start the Prefect server in the background, then the API (0.0.0.0 so both are reachable from outside the container)
prefect server start --host 0.0.0.0 --port 4201 &
uvicorn src.web_service.main:app --host 0.0.0.0 --port 8001
