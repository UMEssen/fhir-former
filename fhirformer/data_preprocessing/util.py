import logging
import random
from fhirformer.fhir.util import check_and_read
from fhirformer.data_preprocessing.data_store import DataStore
from typing import List

logger = logging.getLogger(__name__)


def get_train_val_split(
    patient_ids: List[str], sample_by_letter: List[str] = None, split_ratio: float = 0.8
) -> tuple:
    if sample_by_letter is not None:
        patient_ids = list(set(patient_ids))
        random.shuffle(patient_ids)
        # Split patient IDs into train and validation sets
        split_index = int(split_ratio * len(patient_ids))
        return patient_ids[:split_index], patient_ids[split_index:]
    else:
        train_patients = filter(
            patient_ids,
            lambda x: any(x.startswith(letter) for letter in sample_by_letter),
        )
        val_patients = filter(
            patient_ids,
            lambda x: not any(x.startswith(letter) for letter in sample_by_letter),
        )

        percent_train = len(train_patients) / len(patient_ids)
        percent_val = len(val_patients) / len(patient_ids)
        assert percent_val <= split_ratio, f"Validation set is too large: {percent_val}"
        assert (
            percent_val >= split_ratio - 0.1
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


def print_data_info(pats_int, store_list_global):
    logger.info(f"Overall patients to process {pats_int}")
    logger.info(f"{pats_int} are divided into {len(store_list_global)} lists")
    logger.info(f"Split to patient ratio {pats_int/len(store_list_global)}")


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
