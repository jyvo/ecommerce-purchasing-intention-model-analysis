import joblib
import pandas as pd
from ucimlrepo import fetch_ucirepo

from . import config


def fetch_dataset(force_refresh: bool = False):
    if config.DATASET_CACHE.exists() and not force_refresh:
        cached = joblib.load(config.DATASET_CACHE)
        print(f"Loaded cached dataset from {config.display_path(config.DATASET_CACHE)}")
        return cached["data"], cached["metadata"]

    dataset = fetch_ucirepo(id=config.UCI_DATASET_ID)
    data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"data": data, "metadata": dataset.metadata}, config.DATASET_CACHE)
    print(f"Cached dataset to {config.display_path(config.DATASET_CACHE)}")
    return data, dataset.metadata


def write_raw(data: pd.DataFrame) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(config.RAW_CSV, index=False)
