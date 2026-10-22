<div align="center">

# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](.pre-commit-config.yaml)
</div>

Predict the age of an abalone (number of shell rings, age = rings + 1.5 years) from simple physical measurements, instead of cutting, staining and counting the rings under a microscope.

The project covers the full MLOps cycle: data exploration and experiment tracking (MLflow), training modules orchestrated with Prefect (with a weekly retraining deployment), and a FastAPI prediction API running in Docker.

## 👥 Team

| Name | GitHub |
|---|---|
| Erwan Franco--Tetu | [@erwanfrancotetu](https://github.com/erwanfrancotetu) |
| Licheng Xiong | [@Licheng1004](https://github.com/Licheng1004) |
| Ithier d'Aramon | [@ithierda](https://github.com/ithierda) |
| Bhavesh Chauhan | [@bhaveshchauhan2407](https://github.com/bhaveshchauhan2407) |

## 🚀 Setup

**Prerequisites:** Python 3.11, [uv](https://docs.astral.sh/uv/getting-started/installation/), [Docker Desktop](https://www.docker.com/products/docker-desktop/) and `make`.

1. **Clone the repository**
   ```bash
   git clone https://github.com/ithierda/xhec-mlops-2026-project.git
   cd xhec-mlops-2026-project
   ```

2. **Install the environment** (creates `.venv` with all the dependencies and installs the pre-commit hooks)
   ```bash
   make install
   source .venv/bin/activate # on Windows: .venv\Scripts\activate
   ```
   Without `make`: `uv sync && uv run pre-commit install`

3. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset) and save it as `data/abalone.csv` (the `data/` folder is not versioned).

## ▶️ Run everything

| Command | What it does |
|---|---|
| `make install` | Install the dependencies and the pre-commit hooks |
| `make lint` | Run the code quality checks (same as the CI) |
| `make test` | Run the unit and API tests (also run by the CI) |
| `make mlflow-ui` | Open the MLflow UI to compare the notebook experiments (http://localhost:5000) |
| `make prefect-ui` | Start the Prefect server and UI (http://localhost:4200) |
| `make train` | Train the model with the Prefect flow and save the pickles |
| `make deploy` | Serve the deployment that retrains the model every Monday |
| `make api` | Run the API locally (http://localhost:8000/docs) |
| `make docker-build` | Build the Docker image of the API |
| `make docker-run` | Run the API + Prefect server in Docker |

### 1. Explore the data and track experiments

The notebooks are in `notebooks/` (run them from that folder):
- `eda.ipynb`: data exploration and conclusions
- `modelling.ipynb`: first model (linear regression, test RMSE ≈ 2.2 rings), experiments tracked with MLflow

To compare the MLflow runs: `make mlflow-ui`, then open http://localhost:5000.

### 2. Train the model with Prefect

The training is a Prefect flow (`src/modelling/workflows.py`) made of tasks: load → preprocess → extract features → train → predict → evaluate. It saves the model (`model.pkl`) and the encoder (`dv.pkl`) in `src/web_service/local_objects/`, where the API loads them.

1. **Start the Prefect server and UI** (keep this terminal open):
   ```bash
   make prefect-ui
   ```
   The UI is available at http://localhost:4200.

2. **Train the model** (in a second terminal):
   ```bash
   make train
   ```
   The test RMSE is logged, and the flow run appears in the UI under **Flow Runs**.

3. **Retrain and predict regularly**: serve the two deployments (keep this terminal open):
   ```bash
   make deploy
   ```
   - `Train model/Weekly training` retrains the model every Monday at midnight.
   - `Batch predict/Hourly prediction` predicts the number of rings of every abalone in `data/abalone.csv` every hour with the saved model, and writes them to `data/predictions.csv`.

   Both appear in the UI under **Deployments**, where you can also start a run manually (**Run** → **Quick run**).

### 3. Predict with the API in Docker

The API (`src/web_service/`) loads the model and the encoder, validates the inputs with Pydantic and returns the predicted number of rings. The Docker image runs both the API and a Prefect server (`bin/run_services.sh`).

1. **Build and run the container** (stop `make prefect-ui` first if it is running, as both use port 4200):
   ```bash
   make docker-build
   make docker-run
   ```
   - API documentation: http://localhost:8000/docs
   - Prefect UI: http://localhost:4200

2. **Make a prediction** from the docs page (**POST /predict** → **Try it out**) or with `curl`:
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"sex": "M", "length": 0.455, "diameter": 0.365, "height": 0.095, "whole_weight": 0.514, "shucked_weight": 0.2245, "viscera_weight": 0.101, "shell_weight": 0.15}'
   ```
   Response (status 201):
   ```json
   {"rings_prediction": 9.22}
   ```
   `sex` must be `M`, `F` or `I` (infant) and all measurements must be positive, otherwise the API returns a 422 error.

To run the API without Docker: `make api`, then open http://localhost:8000/docs.

### 4. Run the tests

The tests check that the training code and the API work as expected. They use small fake data built inside the tests, so they don't need the dataset and also run in the CI on every push and pull request.

```bash
make test
```

All 10 tests should pass. The tests are in the `tests/` folder:
- `test_modelling.py`: unit tests for the training code (`src/modelling/`)
- `test_api.py`: tests of the prediction API (`src/web_service/`), sending requests to the app without starting a server

**Training code (`test_modelling.py`)**

| Test | What it checks |
|---|---|
| `test_rename_columns_to_snake_case` | Column names from the raw csv are cleaned up (`Whole weight` → `whole_weight`) |
| `test_filter_outliers_removes_zero_height` | Abalones with a height of 0 (data errors) are removed, the others are kept |
| `test_extract_x_y_one_hot_encodes_sex` | `sex` (M/F/I) is turned into numeric columns the model can use, without losing rows |
| `test_pickle_round_trip` | A saved object (model or encoder) is loaded back exactly as it was |
| `test_evaluate_model_computes_rmse` | The RMSE is 0 for perfect predictions and 2 when every prediction is off by 2 |
| `test_train_model_learns_simple_relation` | The model learns a simple pattern (y = 2x) and predicts it for a new value |

**Prediction API (`test_api.py`)**

| Test | What it checks |
|---|---|
| `test_health_check` | The app starts and `GET /` answers |
| `test_predict_valid_abalone` | A valid abalone gets a positive predicted number of rings (status 201) |
| `test_predict_rejects_unknown_sex` | A `sex` other than M, F or I is rejected (status 422) |
| `test_predict_rejects_negative_measurement` | A negative measurement is rejected (status 422) |

## 📁 Project structure

```
├── .github/workflows/ci.yaml   <- CI: installs the dependencies, runs the pre-commit checks and the tests
├── bin/run_services.sh         <- Starts the Prefect server and the API in the container
├── notebooks/                  <- EDA and modelling notebooks (MLflow tracking)
├── src/
│   ├── modelling/              <- Training: config, preprocessing, training, predicting, Prefect flow and deployment
│   └── web_service/            <- FastAPI app, Pydantic models, inference and the saved model/encoder
├── tests/                      <- Unit tests (training code) and API tests
├── Dockerfile.app              <- Docker image of the API
├── requirements_app.txt        <- Minimal dependencies of the Docker image
├── Makefile                    <- One command per step
└── pyproject.toml / uv.lock    <- Project dependencies (uv)
```

## 💡 Development

- Add a dependency with `uv add <package>==<version>` (updates `pyproject.toml` and `uv.lock`), then `uv sync`.
- The pre-commit hooks format and lint the code (ruff), strip the notebook outputs (nbstripout) and check the lockfile at each commit. Run them on all files with `make lint`.
