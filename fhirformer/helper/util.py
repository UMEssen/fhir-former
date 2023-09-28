import logging
import os
import shutil
import time
from pathlib import Path
from typing import Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


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


def name_from_model(model_name: Union[str, Path]) -> Tuple[str, bool]:
    print(model_name)
    if isinstance(model_name, Path):
        name = model_name.parent.name
        if not model_name.exists():
            raise ValueError(f"Model {model_name} does not exist.")
        load = True
    elif Path(model_name).exists():
        name = Path(model_name).parent.name
        load = True
    else:
        name = model_name.replace("/", "_")
        load = False
    return name, load


def clear_process_data(config):
    if not config["is_live_prediction"]:
        if input("Do you really want to delete the cache (y:yes)?:") != "y":
            exit()
    folders = config["folders_to_clear"]
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                logger.info(f"deleting: {file_path}")
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.info("Failed to delete %s. Reason: %s" % (file_path, e))
