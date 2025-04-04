import logging
import os
import time
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def is_main_process():
    return "LOCAL_RANK" not in os.environ or int(os.environ["LOCAL_RANK"]) == 0


# Time decorator for function execution time measurement
def timed(func):
    def wrapper(*args, **kwargs):
        logger.info(f"Starting {func.__name__}")
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_time = (time.perf_counter() - start) / 60
        logger.info(
            f"Time taken for {func.__name__}: {np.round(elapsed_time,2)} minutes"
        )
        return result

    return wrapper


def get_nondependent_resources(config):
    non_dependent_resources = config["resources_for_task"].get(config["task"], None)
    if non_dependent_resources is None:
        logger.warning(
            f"Task {config['task']} not found in resources_for_task of the config, "
            f"using default value."
        )
        non_dependent_resources = config["resources_for_task"].get("default")
    return non_dependent_resources


def name_from_model(
    model_name: Union[str, Path], roformer: bool = False
) -> Tuple[str, str, bool]:
    if isinstance(model_name, Path):
        name = model_name.parent.parent.name
        plain_name = name.replace("_", "/")
        if not model_name.exists():
            raise ValueError(f"Model {model_name} does not exist.")
        load = True
    elif Path(model_name).exists():
        name = Path(model_name).name
        if name == "best" or name.startswith("checkpoint"):
            name = Path(model_name).parent.parent.name
        plain_name = name.replace("_", "/")
        load = True
    else:
        plain_name = model_name
        name = model_name.replace("/", "_")
        if roformer:
            name = "roformer_" + name
        load = False
    return plain_name, name, load


def get_labels_info(
    labels: list,
    additional_string: str = "",
    stop_after: int = 20,
):
    if isinstance(labels[0], list):
        logger.info(
            f"Average number of labels per sample: {np.mean([len(x) for x in labels]):.2f}"
        )
        labels = [item for sublist in labels for item in sublist]

    counts = list(pd.Series(labels).value_counts().to_dict().items())
    logger.info(
        f"Label counts {'(' + additional_string + ')' if additional_string else ''}"
    )
    if stop_after:
        counts = counts[:stop_after]
    for label, count in counts:
        logger.info(f"{label}: {count}")
