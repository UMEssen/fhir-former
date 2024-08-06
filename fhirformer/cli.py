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
    generate_ds_mortality_samples,
    generate_ds_readmission_samples,
    generate_pre_train_samples,
    sentence_extractor,
)
from fhirformer.fhir import FHIRExtractor, FHIRFilter, FHIRValidator
from fhirformer.helper.util import (
    get_nondependent_resources,
    is_main_process,
    name_from_model,
    timed,
)
from fhirformer.ml import ds_multi_label, ds_single_label, inference, pre_train_llm
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
        "train": ds_single_label.main,
    },
    "ds_main_icd": {
        "generate": generate_ds_main_icd.main,
        "train": ds_single_label.main,
    },
    "ds_mortality": {
        "generate": generate_ds_mortality_samples.main,
        "train": ds_single_label.main,
    },
    "pretrain_fhir_documents": {
        "generate": [generate_pre_train_samples.main, sentence_extractor.main],
        "train": pre_train_llm.main,
    },
    "pretrain_fhir": {
        "generate": generate_pre_train_samples.main,
        "train": pre_train_llm.main,
    },
    "pretrain_documents": {
        "generate": sentence_extractor.main,
        "train": pre_train_llm.main,
    },
}

bool_args = [
    "use_condition",
    "use_procedure",
    "use_imaging_study",
    "use_diagnostic_report",
    "use_biologically_derived_product",
    "use_observation",
    "use_episode_of_care",
    "use_medication",
    "use_service_request",
]


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
    config["task_dir"].mkdir(parents=True, exist_ok=True)
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

    for arg in bool_args:
        parser.add_argument(
            f"--{arg}",
            type=lambda x: (str(x).lower() == "true"),
            default=config.get(arg, False),
        )

    parser.add_argument("--task", type=str, default=config["task"], required=True)
    parser.add_argument("--debug", action="store_true", default=config["debug"])
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
    parser.add_argument(
        "--max_train_samples", type=int, default=config["max_train_samples"]
    )
    parser.add_argument("--run_name", type=str, default=None)

    args = parser.parse_args()

    # Log parsed arguments
    logger.info(f"Parsed arguments: {vars(args)}")

    return args


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

    # check if any of the use args are set to true
    if any(config.get(arg) for arg in bool_args):
        data_id_components = [arg for arg in bool_args if config.get(arg)]
        data_id_components = [comp[4:] for comp in data_id_components]
        config["data_id"][config["task"]] = "+".join(data_id_components)

    # Create model folder
    config["model"], config["model_name"], config["loaded_model"] = name_from_model(
        config["model_checkpoint"], config["use_roformer"]
    )

    config["task_dir"] = config["root_dir"] / config["task"]

    config["sample_dir"] = (
        config["task_dir"] / f"sampled_{config['data_id'][config['task']]}"
    )

    if config["step"] == "all":
        config["step"] = "data+sampling+train+test"

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
        if isinstance(pipelines[config["task"]]["generate"], list):
            for pipeline in pipelines[config["task"]]["generate"]:
                pipeline(config)
        else:
            pipelines[config["task"]]["generate"](config)
        with (config["task_dir"] / "config_sampling.pkl").open("wb") as of:
            pickle.dump(config, of)

    if "train" in config["step"] or "test" in config["step"]:
        init_wandb(config)

    if "train" in config["step"]:
        german_tz = pytz.timezone("Europe/Berlin")
        current_time_german = datetime.now(german_tz).strftime("%Y%m%d_%H_%M")

        config["model_dir"] = (
            config["task_dir"]
            / config["model_name"]
            / (
                current_time_german
                + "_"
                + (wandb.run.id if not config["debug"] else "debug")
            )
        )

        pipelines[config["task"]]["train"](config)
        with (config["task_dir"] / "config_train.pkl").open("wb") as of:
            pickle.dump(config, of)

        # Set the model checkpoint to model dir such that it can be used for inference
        config["model_checkpoint"] = config["model_dir"] / "best"

    if "test" in config["step"]:
        inference.main(config)
        with (config["task_dir"] / "config_test.pkl").open("wb") as of:
            pickle.dump(config, of)
