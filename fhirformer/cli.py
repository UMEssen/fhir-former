import argparse
import logging
import pickle
from datetime import datetime
from pathlib import Path

import yaml

from fhirformer.data_preprocessing import (
    generate_ds_icd_samples,
    generate_ds_image_samples,
    generate_ds_main_icd,
    generate_pre_train_samples,
)
from fhirformer.fhir import FHIRExtractor, FHIRFilter, FHIRValidator
from fhirformer.helper.util import get_nondependent_resources, name_from_model, timed
from fhirformer.ml import ds_main_diag_llm, ds_multi_label, pre_train_llm

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


# Pipeline functions
@timed
def run_pipeline(*tasks, config=None):
    for task in tasks:
        task(config)


@timed
def build_cache(config):
    logger.info(
        f"Extracting data between {config['start_datetime']} and {config['end_datetime']}."
    )
    extract = FHIRExtractor(config)
    filt = FHIRFilter(config)
    validator = FHIRValidator(config)

    non_dependent_resources = get_nondependent_resources(config)

    dependent_resources = sorted(
        ["encounter", "patient"]
    )  # todo think about merging encounters with based basedon Encounter/95e3c9cb3f4b5bd691b9a426096506c44ce070cd63d8643fe572ab7c5d844c2a
    # dependent_resources = []
    logger.info(
        f"The following resources will be computed: "
        f"{dependent_resources + non_dependent_resources}"
    )

    # encounter need to be build first
    for resource in dependent_resources + non_dependent_resources:
        logger.info(f"Extracting {resource}...")
        extract.build(resource)

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
        default="LennartKeller/longformer-gottbert-base-8192-aw512",
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
        default="pretrain",
        required=True,
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["train", "test"],
        default="train",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=config["debug"],
    )
    parser.add_argument(
        "--download_documents",
        type=bool,
        default=config["download_documents"],
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

    # if config["debug"]:
    #     config["start_datetime"] = "2021-03-01"
    #     config["end_datetime"] = "2021-03-30"
    #     config["run_id"] = "testing_30d"
    #     config.update(vars(args))

    config["root_dir"] = config["root_dir"] / config["run_id"]
    config["data_dir"] = config["root_dir"] / "data_raw"
    config["data_dir"].mkdir(parents=True, exist_ok=True)

    # Create model folder
    config["model_name"], config["loaded_model"] = name_from_model(
        config["model_checkpoint"]
    )

    config["task_dir"] = config["root_dir"] / config["task"]
    config["model_dir"] = (
        config["task_dir"]
        / config["model_name"]
        / datetime.now().strftime("%Y%m%d_%H_%M")
    )
    config["model_dir"].mkdir(parents=True, exist_ok=True)

    logger.info(f"The outputs will be stored in {config['task_dir']}.")

    build_cache(config)

    if config["download_documents"]:
        exit()

    with (config["task_dir"] / "config.pkl").open("wb") as of:
        pickle.dump(config, of)

    if args.task == "ds_icd":  # expanding labels per encounter
        # todo debug generate_ds_icd_samples.main
        run_pipeline(
            generate_ds_icd_samples.main,
            ds_multi_label.main,
            config=config,
        )
    elif args.task == "ds_image":
        # todo train some more
        run_pipeline(
            generate_ds_image_samples.main,
            ds_multi_label.main,
            config=config,
        )
    elif args.task == "ds_main_icd":
        # todo debug ds_main_diag_samples.main + create ds_single_label.main for generic training with single label
        run_pipeline(
            generate_ds_main_icd.main,
            ds_main_diag_llm.main,
            config=config,
        )
    elif args.task in {"pretrain_fhir_documents", "pretrain_fhir"}:
        # todo train on 5y data
        run_pipeline(
            generate_pre_train_samples.main,
            pre_train_llm.main,
            config=config,
        )
    elif args.task == "pretrain_documents":
        run_pipeline(
            pre_train_llm.main,
            config=config,
        )
    else:
        raise ValueError(f"Task {args.task} not recognized")
