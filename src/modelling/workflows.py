from config import MODELS_DIRPATH, RANDOM_STATE, TEST_SIZE
from loguru import logger
from predicting import predict
from prefect import flow
from preprocessing import extract_x_y, load_data, preprocess_data
from sklearn.model_selection import train_test_split
from training import evaluate_model, train_model
from utils import load_pickle, pickle_object


@flow(name="Train model")
def train_model_workflow(trainset_path: str, artifacts_dirpath: str = str(MODELS_DIRPATH)) -> float:
    """Train a model on the data at the given path, save the model and the DictVectorizer, and return the test RMSE."""
    df = load_data(trainset_path)
    df = preprocess_data(df)
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    x_train, y_train, dv = extract_x_y(train_df)
    x_test, y_test, _ = extract_x_y(test_df, dv=dv)

    model = train_model(x_train, y_train)
    rmse = evaluate_model(y_test, predict(x_test, model))
    logger.info(f"Test RMSE: {rmse:.3f}")

    pickle_object(model, f"{artifacts_dirpath}/model.pkl")
    pickle_object(dv, f"{artifacts_dirpath}/dv.pkl")
    logger.info(f"Model and DictVectorizer saved in {artifacts_dirpath}")
    return rmse


@flow(name="Batch predict")
def batch_predict_workflow(input_path: str, output_path: str, artifacts_dirpath: str = str(MODELS_DIRPATH)) -> None:
    """Predict the number of rings of every abalone in a csv file with the saved model, and save the predictions."""
    model = load_pickle(f"{artifacts_dirpath}/model.pkl")
    dv = load_pickle(f"{artifacts_dirpath}/dv.pkl")

    raw_df = load_data(input_path)
    df = preprocess_data(raw_df)
    logger.info(f"{len(raw_df) - len(df)} abalones with a height of 0 removed (no prediction for them)")
    x, _, _ = extract_x_y(df, dv=dv, with_target=False)
    df["rings_prediction"] = predict(x, model)

    df.to_csv(output_path, index=False)
    logger.info(f"{len(df)} predictions saved in {output_path}")
