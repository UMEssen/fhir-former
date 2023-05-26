""" Main file where one can define what functions to run """
import argparse
import logging

# Imports
import time
import warnings
from pathlib import Path

import yaml
from colorlog import ColoredFormatter

from app.data_preprocessing import (
    cache_builder,
    generate_ds_icd_samples,
    generate_pre_train_samples,
)
from app.helper import data_visualization
from app.ml import ds_train_llm, pre_train_llm

logger = logging.getLogger(__name__)
LOG_LEVEL = logging.INFO
logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(
    "%(log_color)s[%(levelname).1s %(log_color)s%(asctime)s] - %(log_color)s%(name)s %(reset)s- "
    "%(message)s"
)
external_loggers = [logger]
for package in ["fhir_pyrate", "urllib3"]:
    package_logger = logging.getLogger(package)
    package_logger.setLevel(LOG_LEVEL)
    external_loggers.append(package_logger)


# config = yaml.safe_load((Path.cwd() / "app/config/config.yaml").open())
config = yaml.safe_load((Path.cwd() / "app/config/config_training.yaml").open())


def set_paths() -> None:
    def helper_set_paths(folder_name: str) -> str:
        return config["root_dir"] / Path(folder_name)

    for key, value in config.items():
        if isinstance(value, str) and value.__contains__("./"):
            config[key] = helper_set_paths(value)
        if isinstance(value, list):
            if value[0].__contains__("./"):
                config[key] = [helper_set_paths(x) for x in config["folders_to_clear"]]
    return


# FHIR
def build_cache():
    logger.info(f"Checking and building cache...")
    start = time.time()
    cache_builder.main(config)
    logger.info(f"Time taken: {(time.time() - start) / 60} minutes")


def preprocess_resources_pre_train():
    logger.info(f"Starting model training")
    start = time.time()
    generate_pre_train_samples.main(config)
    logger.info(f"Time taken: {(time.time() - start) / 60} minutes")


def preprocess_resources_ds():
    logger.info(f"Starting model training")
    start = time.time()
    generate_ds_icd_samples.main(config)
    logger.info(f"Time taken: {(time.time() - start) / 60} minutes")


def launch_visualization():
    logger.info(f"Starting model visualization")
    start = time.time()
    data_visualization.main(config)
    logger.info(f"Time taken: {(time.time() - start) / 60} minutes")


def launch_training():
    logger.info(f"Starting model training")
    start = time.time()
    pre_train_llm.main(config)
    logger.info(f"Time taken: {(time.time() - start) / 60} minutes")


def launch_ds_training():
    logger.info(f"Starting model training on downstream task")
    start = time.time()
    ds_train_llm.main(config)
    logger.info(f"Time taken: {(time.time() - start) / 60} minutes")


def train_publish_pipeline():
    blocks = [
        set_paths,
        build_cache,
        # todo live predict sth
    ]
    start = time.time()
    for block in blocks:
        block()
    end = time.time()
    logger.info(f"Process done. Time taken: {(end - start) / 60} minutes")


def pre_train_pipeline():
    blocks = [
        set_paths,
        # TODO use fhir metrics to pull data for now we pretend the data is already there
        build_cache,
        preprocess_resources_pre_train,
        launch_training,
    ]
    start = time.time()
    for block in blocks:
        block()
    end = time.time()
    logger.info(f"Process done. Time taken: {(end - start) / 60} minutes")


def ds_train_pipeline():
    blocks = [
        set_paths,
        build_cache,
        preprocess_resources_ds,
        launch_ds_training,
    ]
    start = time.time()
    for block in blocks:
        block()
    end = time.time()
    logger.info(f"Process done. Time taken: {(end - start) / 60} minutes")


def visualize_pipeline():
    blocks = [
        set_paths,
        # build_cache,
        # preprocess_resources,
        launch_visualization,
    ]
    start = time.time()
    for block in blocks:
        block()
    end = time.time()
    logger.info(f"Process done. Time taken: {(end - start) / 60} minutes")


def train_publish_pipline_controller():
    config["bdp_params"].update({"shipProductCode": "TKZ,TKP"})
    train_publish_pipeline()


def parse_args_local() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", type=Path, required=False, default=config["root_dir"]
    )

    parser.add_argument(
        "--is_live_pipline",
        type=lambda x: (str(x).lower() == "true"),
        required=False,
        default=config["is_live_prediction"],
    )

    parser.add_argument(
        "--wandb_log",
        type=lambda x: (str(x).lower() == "true"),
        required=False,
        default=False,
    )
    parser.add_argument(
        "--is_ds",
        type=lambda x: (str(x).lower() == "true"),
        required=False,
        default=False,
    )
    args = parser.parse_args()
    config.update(vars(args))


if __name__ == "__main__":
    parse_args_local()

    if config["is_live_prediction"]:
        train_publish_pipline_controller()
    elif config["is_ds"]:
        ds_train_pipeline()
    else:
        pre_train_pipeline()
        # visualize_pipeline()
