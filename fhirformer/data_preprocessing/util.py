import logging

from fhirformer.fhir.util import check_and_read
from fhirformer.data_preprocessing.data_store import DataStore
from typing import List
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


def load_datastore(datastore_path: Path):
    with datastore_path.open("rb") as f:
        datastore = pickle.load(f)
    return datastore


def get_valid_labels(path: str, column: str, percentual_cutoff: float = 0.005) -> list:
    resource = check_and_read(path)
    codes = resource[column].value_counts(normalize=True)
    logger.info(f"Number of unique codes: {len(codes)}")
    filtered_codes = codes[codes > percentual_cutoff].index.tolist()
    logger.info(f"Number of unique codes after filtering: {len(filtered_codes)}")
    return filtered_codes


def validate_resources(resources, config):
    for resource in resources:
        if resource not in config["text_sampling_column_maps"].keys():
            raise NotImplementedError(
                f"Resource {resource} not in config['text_sampling_column_maps'].keys()."
            )


def get_column_map_txt_resources(config, resources_for_pre_training):
    return {
        k: v
        for k, v in config["text_sampling_column_maps"].items()
        if k in resources_for_pre_training
    }


def get_patient_ids_lists(store_list: List["DataStore"]):
    return [
        store_global.patient_df["patient_id"].unique().tolist()
        for store_global in store_list
    ]


def skip_build(config: dict) -> bool:
    if (
        not config["rerun_cache"]
        and (config["task_dir"] / "train.json").exists()
        and (config["task_dir"] / "test.json").exists()
        and not config["debug"]
    ):
        logger.info("Skipping sampling ...")
        return True
    else:
        return False
