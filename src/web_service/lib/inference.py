from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from src.web_service.lib.models import InputData


def run_inference(input_data: InputData, dv: DictVectorizer, model: LinearRegression) -> float:
    """Predict the number of rings of one abalone.

    The input fields have the same names as the columns used for training (snake_case),
    so the fitted DictVectorizer can encode them directly (one-hot encoding of `sex`).
    """
    x = dv.transform([input_data.dict()])
    return float(model.predict(x)[0])
