import argparse
import logging
import time
from pathlib import Path
import yaml
from colorlog import ColoredFormatter

from app.data_preprocessing import (
    cache_builder,
    generate_ds_icd_samples,
    generate_pre_train_samples,
    generate_ds_imgage_samples,
)
from app.helper import data_visualization
from app.ml import ds_icd_llm, pre_train_llm, ds_main_diag_llm

# Set up logging
logger = logging.getLogger(__name__)
LOG_LEVEL = logging.INFO
logging.root.setLevel(LOG_LEVEL)


def setup_logging():
    formatter = ColoredFormatter(
        "%(log_color)s[%(levelname).1s %(log_color)s%(asctime)s] - %(log_color)s%(name)s %(reset)s- "
        "%(message)s"
    )
    for package in ["fhir_pyrate", "urllib3", __name__]:
        package_logger = logging.getLogger(package)
        package_logger.setLevel(LOG_LEVEL)


# Load config file
def load_config(file_path: str = "app/config/config_training.yaml"):
    return yaml.safe_load((Path.cwd() / file_path).open())


config = load_config()


# Set paths in config
def set_paths() -> None:
    def helper_set_paths(folder_name: str) -> str:
        # if a task is defined in the args, use the task folder except for the encounter.ftr file
        if (
            args.task
            and folder_name.startswith("./data_")
            and not folder_name.startswith("./data_raw/encounter.ftr")
            and not folder_name.startswith("./data_raw/condition.ftr")
        ):
            return config["root_dir"] / Path(args.task) / Path(folder_name)
        elif folder_name.startswith("./model_dir"):
            return config["root_dir"] / Path(args.task) / Path("models")
        return config["root_dir"] / Path(folder_name)

    for key, value in config.items():
        if isinstance(value, str) and "./" in value:
            config[key] = helper_set_paths(value)
        elif isinstance(value, list) and "./" in value[0]:
            config[key] = [helper_set_paths(x) for x in config["folders_to_clear"]]


# Time decorator for function execution time measurement
def timed(func):
    def wrapper(*args, **kwargs):
        logger.info(f"Starting {func.__name__}")
        start = time.time()
        result = func(*args, **kwargs)
        elapsed_time = (time.time() - start) / 60
        logger.info(f"Time taken for {func.__name__}: {elapsed_time} minutes")
        return result

    return wrapper


# Functions with time measurement
@timed
def build_cache():
    cache_builder.main(config)


@timed
def preprocess_resources_pre_train():
    generate_pre_train_samples.main(config)


@timed
def preprocess_resources_ds_icd():
    generate_ds_icd_samples.main(config)


@timed
def preprocess_resources_ds_img():
    generate_ds_imgage_samples.main(config)


@timed
def launch_visualization():
    data_visualization.main(config)


@timed
def launch_training():
    pre_train_llm.main(config)


@timed
def launch_ds_training():
    ds_icd_llm.main(config)


# Pipeline functions
def run_pipeline(*blocks):
    set_paths()
    start = time.time()
    for block in blocks:
        block()
    end = time.time()
    logger.info(f"Process done. Time taken: {(end - start) / 60} minutes")


def train_publish_pipeline():
    run_pipeline(build_cache)


def pre_train_pipeline():
    run_pipeline(build_cache, preprocess_resources_pre_train, launch_training)


def ds_icd_predict():
    run_pipeline(preprocess_resources_ds_icd, launch_ds_training)


def ds_image_predict():
    run_pipeline(build_cache, preprocess_resources_ds_img, launch_ds_training)


def visualize_pipeline():
    run_pipeline(launch_visualization)


def train_publish_pipline_controller():
    config["bdp_params"].update({"shipProductCode": "TKZ,TKP"})
    train_publish_pipeline()


def parse_args_local() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=Path, default=config["root_dir"])
    parser.add_argument(
        "--is_live_pipline",
        type=lambda x: str(x).lower() == "true",
        default=config["is_live_prediction"],
    )
    parser.add_argument(
        "--wandb_log", type=lambda x: str(x).lower() == "true", default=False
    )
    parser.add_argument("--artifact", type=str, default=False)

    subparsers = parser.add_subparsers(
        dest="task", help="start training for downstream task icd prediction"
    )

    # DS TASK ICD PREDICTION
    parser_ds_icd = subparsers.add_parser(
        "ds_icd", help="start training for downstream task icd prediction"
    )
    parser_ds_icd.add_argument(
        "--is_ds", type=lambda x: str(x).lower() == "true", default=False
    )

    # DS TASK IMAGE PREDICTION
    parser_ds_image = subparsers.add_parser(
        "ds_image", help="start training for downstream task radio image prediction"
    )
    parser_ds_image.add_argument(
        "--is_ds", type=lambda x: str(x).lower() == "true", default=False
    )

    ds_main_diag = subparsers.add_parser(
        "ds_main_diag", help="predict icd codes for downstream task icd prediction"
    )

    args = parser.parse_args()
    config.update(vars(args))
    return args


if __name__ == "__main__":
    setup_logging()
    args = parse_args_local()
    if config["is_live_prediction"]:
        # todo to be implemented
        train_publish_pipline_controller()
    elif args.task == "ds_icd":
        ds_icd_predict()
    elif args.task == "ds_image":
        ds_image_predict()
    elif args.task == "ds_main_diag":
        set_paths()
        ds_main_diag_llm.main(config)
    else:
        pre_train_pipeline()
        # visualize_pipeline()
