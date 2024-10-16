import logging
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import List

from fhirformer.data_preprocessing.data_store import DataStore
from fhirformer.fhir.util import check_and_read

logger = logging.getLogger(__name__)


def load_datastore(datastore_path: Path):
    with datastore_path.open("rb") as f:
        datastore = pickle.load(f)
    return datastore


def get_train_val_split(
    patient_data: List[tuple],  # Each tuple contains (patient_id, labels)
    split_ratio: float = 0.8,
    tolerance: float = 0.3,
) -> tuple:
    # Dictionary to track which labels are associated with which patient_ids
    label_to_patients = defaultdict(set)
    for patient_id, labels in patient_data:
        for label in labels:
            label_to_patients[label].add(patient_id)

    # Ensure all labels in test are also in train
    patient_ids = list(set([patient[0] for patient in patient_data]))
    random.seed(42)
    random.shuffle(patient_ids)

    # Initialize train and validation sets
    train_patients = set()
    val_patients = set(patient_ids)

    # Ensure at least one patient for each label is in the training set
    for _label, patients_with_label in label_to_patients.items():
        if not patients_with_label.intersection(train_patients):
            # Move at least one patient with this label to the train set
            patient_to_move = patients_with_label.pop()
            train_patients.add(patient_to_move)
            if patient_to_move in val_patients:
                val_patients.remove(patient_to_move)

    # Balance remaining patients according to the split ratio
    remaining_patients = list(val_patients)
    random.shuffle(remaining_patients)
    num_train_to_add = int(split_ratio * len(patient_ids)) - len(train_patients)

    # Add remaining patients to train until we reach the split ratio
    train_patients.update(remaining_patients[:num_train_to_add])
    val_patients = set(remaining_patients[num_train_to_add:])

    # Ensure the split ratio is respected with tolerance
    percent_train = len(train_patients) / len(patient_ids)
    percent_val = len(val_patients) / len(patient_ids)
    assert (
        percent_train <= split_ratio + tolerance
    ), f"Training set is too large: {percent_train:.2f}"
    assert (
        percent_val <= (1 - split_ratio) + tolerance
    ), f"Validation set is too large: {percent_val:.2f}"

    return list(train_patients), list(val_patients)


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
        and (config["sample_dir"] / "train.jsonl").exists()
        and (config["sample_dir"] / "test.jsonl").exists()
        and not config["debug"]
    ):
        logger.info(
            f"Skipping sampling as it already exists in {config['sample_dir']}."
        )
        return True
    else:
        return False
