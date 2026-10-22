from config import DATA_DIRPATH, MODELS_DIRPATH
from prefect import serve
from workflows import batch_predict_workflow, train_model_workflow

if __name__ == "__main__":
    train_deployment = train_model_workflow.to_deployment(
        name="Weekly training",
        version="0.1.0",
        tags=["training"],
        cron="0 0 * * 1",  # every Monday at midnight
        parameters={
            "trainset_path": str(DATA_DIRPATH / "abalone.csv"),
            "artifacts_dirpath": str(MODELS_DIRPATH),
        },
    )

    predict_deployment = batch_predict_workflow.to_deployment(
        name="Hourly prediction",
        version="0.1.0",
        tags=["prediction"],
        interval=3600,  # every hour
        parameters={
            "input_path": str(DATA_DIRPATH / "abalone.csv"),
            "output_path": str(DATA_DIRPATH / "predictions.csv"),
        },
    )

    serve(train_deployment, predict_deployment)
