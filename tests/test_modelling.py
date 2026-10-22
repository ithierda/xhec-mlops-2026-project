from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from preprocessing import extract_x_y, filter_outliers, rename_columns
from training import evaluate_model, train_model
from utils import load_pickle, pickle_object


@pytest.fixture
def abalones() -> pd.DataFrame:
    """Three abalones (one of each sex) with the columns already renamed, as after preprocessing."""
    return pd.DataFrame(
        {
            "sex": ["M", "F", "I"],
            "length": [0.45, 0.53, 0.33],
            "diameter": [0.36, 0.42, 0.25],
            "height": [0.09, 0.14, 0.08],
            "whole_weight": [0.51, 0.68, 0.20],
            "shucked_weight": [0.22, 0.26, 0.09],
            "viscera_weight": [0.10, 0.14, 0.04],
            "shell_weight": [0.15, 0.21, 0.06],
            "rings": [15, 9, 7],
        }
    )


def test_rename_columns_to_snake_case() -> None:
    """Column names from the raw csv are turned into snake_case."""
    df = pd.DataFrame(columns=["Sex", "Whole weight", "Rings"])
    assert list(rename_columns(df).columns) == ["sex", "whole_weight", "rings"]


def test_filter_outliers_removes_zero_height(abalones: pd.DataFrame) -> None:
    """Abalones with a height of 0 are removed, the others are kept."""
    abalones.loc[0, "height"] = 0
    filtered = filter_outliers(abalones)
    assert len(filtered) == 2
    assert (filtered["height"] > 0).all()


def test_extract_x_y_one_hot_encodes_sex(abalones: pd.DataFrame) -> None:
    """The DictVectorizer creates one column per sex and keeps one row per abalone."""
    x, y, dv = extract_x_y.fn(abalones)
    assert {"sex=M", "sex=F", "sex=I"} <= set(dv.get_feature_names_out())
    assert x.shape[0] == len(abalones)
    assert list(y) == [15, 9, 7]


def test_pickle_round_trip(tmp_path: Path) -> None:
    """An object saved with pickle_object is loaded back identical with load_pickle."""
    path = tmp_path / "object.pkl"
    pickle_object({"a": 1}, path)
    assert load_pickle(path) == {"a": 1}


def test_evaluate_model_computes_rmse() -> None:
    """The RMSE is 0 for perfect predictions and 2 when every prediction is off by 2."""
    y_true = np.array([5.0, 10.0, 15.0])
    assert evaluate_model.fn(y_true, y_true) == 0
    assert evaluate_model.fn(y_true, y_true + 2) == pytest.approx(2)


def test_train_model_learns_simple_relation() -> None:
    """The model learns y = 2x and predicts it for a new value."""
    x = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = 2 * x.ravel()
    model = train_model.fn(x, y)
    assert model.predict([[10.0]])[0] == pytest.approx(20)
