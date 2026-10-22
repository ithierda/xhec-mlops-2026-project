from pathlib import Path

# MODELS
MODEL_VERSION = "0.0.1"
LOCAL_OBJECTS_DIRPATH = Path(__file__).resolve().parent / "local_objects"
PATH_TO_MODEL = LOCAL_OBJECTS_DIRPATH / "model.pkl"
PATH_TO_DV = LOCAL_OBJECTS_DIRPATH / "dv.pkl"

# MISC
APP_TITLE = "AbaloneAgePredictionApp"
APP_DESCRIPTION = (
    "A simple API to predict the age of an abalone (number of rings) from its physical measurements: "
    "sex, length, diameter, height and weights."
)
APP_VERSION = "0.0.1"
