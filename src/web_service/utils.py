import pickle
from pathlib import Path
from typing import Any


def load_object(path: Path) -> Any:
    """Load a pickled object (model or DictVectorizer)."""
    with Path(path).open("rb") as f:
        return pickle.load(f)
