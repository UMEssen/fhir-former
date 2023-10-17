import argparse
import logging
import pickle
from datetime import datetime
from pathlib import Path
import pytz

import wandb
import yaml

from fhirformer.data_preprocessing import (
    generate_ds_icd_samples,
    generate_ds_image_samples,
    generate_ds_main_icd,
    generate_ds_readmission_samples,
    generate_pre_train_samples,
)
from fhirformer.fhir import FHIRExtractor, FHIRFilter, FHIRValidator
from fhirformer.helper.util import (
    get_nondependent_resources,
    is_main_process,
    name_from_model,
    timed,
)
from fhirformer.ml import ds_multi_label, ds_single_label, pre_train_llm
from fhirformer.ml.util import init_wandb

pipelines = {
    "ds_icd": {
        "generate": generate_ds_icd_samples.main,
        "train": ds_multi_label.main,
    },
    "ds_image": {
        "generate": generate_ds_image_samples.main,
        "train": ds_multi_label.main,
    },
    "ds_readmission": {
        "generate": generate_ds_readmission_samples.main,
        # todo sampling is done now train whenever (ds_single_label class ready)...
        "train": ds_single_label.main,
    },
    "ds_main_icd": {
        "generate": generate_ds_main_icd.main,
        "train": ds_single_label.main,
    },
    "pretrain_fhir_documents": {
        "generate": generate_pre_train_samples.main,
        "train": pre_train_llm.main,
    },
    "pretrain_fhir": {
        "generate": generate_pre_train_samples.main,
        "train": pre_train_llm.main,
    },
    "pretrain_documents": {
        "generate": lambda x: x,
        "train": pre_train_llm.main,
    },
}


# Set up logging
LOG_LEVEL = logging.INFO
logging.basicConfig(
    format="%(levelname)s %(asctime)s [%(name)s.%(funcName)s:%(lineno)d]: %(message)s",
    level=LOG_LEVEL,
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logging.getLogger().setLevel(LOG_LEVEL)

logger = logging.getLogger(__name__)


# Load config file
def load_config(file_path: str = "fhirformer/config/config_training.yaml"):
    return yaml.safe_load((Path.cwd() / file_path).open())


@timed
def build_cache(config):
    logger.info(
        f"Extracting data between {config['start_datetime']} and {config['end_datetime']}."
    )
    extract = FHIRExtractor(config)
    filt = FHIRFilter(config)
    validator = FHIRValidator(config)

    non_dependent_resources = get_nondependent_resources(config)

    # TODO: think about merging encounters with based basedon
    #  Encounter/95e3c9cb3f4b5bd691b9a426096506c44ce070cd63d8643fe572ab7c5d844c2a
    dependent_resources = sorted(["encounter", "patient"])
    # dependent_resources = []
    logger.info(
        f"The following resources will be computed: "
        f"{dependent_resources + non_dependent_resources}"
    )

    # encounter need to be build first
    for resource in dependent_resources + non_dependent_resources:
        logger.info(f"Extracting {resource}...")
        extract.build(resource)

    if config["task"] == "None":
        logger.info("Skipping filtering and validation because task is none.")
        return
    # filter patients needs to run first as we filter patients based on encounters patient_ids
    for resource in sorted(dependent_resources, reverse=True) + non_dependent_resources:
        logger.info(f"Filtering {resource}...")
        filt.filter(resource)
    for resource in dependent_resources + non_dependent_resources:
        logger.info(f"Validating {resource}...")
        validator.validate(resource)


def parse_args_local(config) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=Path, default=config["root_dir"])
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default=config["model_checkpoint"],
        help="Path to trained model or huggingface model name as string",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--task",
        type=str,
        default=config["task"],
        required=True,
    )
    # TODO: Currently not used, we only do training
    parser.add_argument(
        "--phase",
        type=str,
        choices=["train", "test"],
        default="train",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=config["debug"],
    )
    parser.add_argument(
        "--download_documents",
        action="store_true",
        default=config["download_documents"],
    )
    parser.add_argument(
        "--step",
        default=config["step"],
        help="Plus separated list steps to run. Valid options: data, sampling, train, all.",
    )

    return parser.parse_args()


def run():
    config = load_config()

    args = parse_args_local(config)
    config.update(vars(args))
    if config["debug"]:
        logger.warning(
            "WARNING!!! You are running fhirformer in debug mode, "
            "please change this when you are done testing."
        )

    config["root_dir"] = config["root_dir"] / config["run_id"]
    config["data_dir"] = config["root_dir"] / "data_raw"
    config["data_dir"].mkdir(parents=True, exist_ok=True)

    # Create model folder
    config["model"], config["model_name"], config["loaded_model"] = name_from_model(
        config["model_checkpoint"], config["use_roformer"]
    )

    config["task_dir"] = config["root_dir"] / config["task"]

    if config["step"] == "all":
        config["step"] = "data+sampling+train"

    config["step"] = config["step"].split("+")

    if "data" in config["step"] and is_main_process():
        build_cache(config)
        with (config["task_dir"] / "config_data.pkl").open("wb") as of:
            pickle.dump(config, of)
        if config["download_documents"]:
            exit()

    assert config["task"] in pipelines, f"Task {config['task']} not found."

    config["task_dir"].mkdir(parents=True, exist_ok=True)
    logger.info(f"The outputs will be stored in {config['task_dir']}.")

    if "sampling" in config["step"] and is_main_process():
        pipelines[config["task"]]["generate"](config)
        with (config["task_dir"] / "config_sampling.pkl").open("wb") as of:
            pickle.dump(config, of)

    if "train" in config["step"]:
        init_wandb(config)
        german_tz = pytz.timezone("Europe/Berlin")
        current_time_german = datetime.now(german_tz).strftime("%Y%m%d_%H_%M")

        config["model_dir"] = (
            config["task_dir"]
            / config["model_name"]
            / (current_time_german + "_" + wandb.run.id)
        )
        pipelines[config["task"]]["train"](config)
        with (config["task_dir"] / "config_train.pkl").open("wb") as of:
            pickle.dump(config, of)
