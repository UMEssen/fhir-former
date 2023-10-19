import logging
import pickle
import random
from pathlib import Path
from typing import List

import pandas as pd

from fhirformer.data_preprocessing.data_store import DataStore
from fhirformer.fhir.util import check_and_read

logger = logging.getLogger(__name__)


def load_datastore(datastore_path: Path):
    with datastore_path.open("rb") as f:
        datastore = pickle.load(f)
    return datastore


def get_train_val_split(
    patient_ids: List[str],
    sample_by_letter: List[str] = None,
    split_ratio: float = 0.8,
    tolerance: float = 0.3,
) -> tuple:
    patient_ids = list(set(patient_ids))
    if sample_by_letter is None:
        random.seed(42)
        logger.info(f"Splitting the patients using {split_ratio} split ratio.")
        random.shuffle(patient_ids)
        # Split patient IDs into train and validation sets
        split_index = int(split_ratio * len(patient_ids))
        return patient_ids[:split_index], patient_ids[split_index:]
    else:
        logger.info(
            f"Splitting the patients using {sample_by_letter}. "
            f"The patients starting with these characters will be in the validation set."
        )
        patient_series = pd.Series(patient_ids)
        train_patients = patient_series[
            ~patient_series.str.startswith(tuple(sample_by_letter))
        ].tolist()
        val_patients = patient_series[~patient_series.isin(train_patients)].tolist()
        percent_train = len(train_patients) / len(patient_ids)
        percent_val = len(val_patients) / len(patient_ids)
        logger.info(
            f"Train set: {len(train_patients)} ({percent_train:.2f}), "
            f"Validation set: {len(val_patients)} ({percent_val:.2f})"
        )
        assert percent_val <= split_ratio, f"Validation set is too large: {percent_val}"
        assert (
            percent_train <= split_ratio + tolerance
        ), f"Training set is too large: {percent_train}"

        return train_patients, val_patients


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
        and (config["task_dir"] / "sampled" / "train.jsonl").exists()
        and (config["task_dir"] / "sampled" / "test.jsonl").exists()
        and not config['debug']
    ):
        logger.info(f"Skipping sampling as {config['task_dir']}/sampled/ already exists.")
        return True
    else:
        return False
