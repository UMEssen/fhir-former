import os
from typing import Tuple

import numpy as np
import wandb
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit


def get_param_for_task_model(config, param: str, task: str, model: str):
    if task in config[param]:
        if isinstance(config[param][task], dict) and model in config[param][task]:
            return config[param][task][model]
        else:
            return config[param][task]["default"]
    return config[param]["default"]


def split_dataset(
    dataset: Dataset, train_ratio: float = 0.8
) -> Tuple[Dataset, Dataset]:
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = dataset_size - train_size

    # Split the dataset into training and validation sets
    # TODO: Could also made this stratified, it would be better
    splitter = GroupShuffleSplit(test_size=val_size, n_splits=2, random_state=42)
    split = splitter.split(dataset, groups=dataset["patient_id"])
    train_inds, val_inds = next(split)

    return dataset.select(train_inds), dataset.select(val_inds)


def init_wandb(config):
    project_name = (
        "fhirformer_ds"
        if config["task"].startswith("ds_")
        else "fhirformer_pretraining"
    )

    wandb.init(
        tags=["baseline"],
        project=project_name,
        name=config["model_name"] + "_" + config["run_id"],
        mode="disabled" if config["debug"] else "online",
        entity="ship-ai-autopilot",
        group=config["task"].split("_")[1],
        resume=config["run_name"] if config["model_checkpoint"] else None,
    )
    wandb.run.log_code(".")


def get_evaluation_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    single_label=False,
    zero_division=0,
):
    if single_label:
        return {
            "accuracy": (predictions == labels).mean(),
            "precision": precision_score(
                labels,
                predictions,
                zero_division=zero_division,
            ),
            "recall": recall_score(
                labels,
                predictions,
                zero_division=zero_division,
            ),
            "f1": f1_score(
                labels,
                predictions,
                zero_division=zero_division,
            ),
        }
    else:
        return {
            "accuracy": (predictions == labels).mean(),
            "macro_precision": precision_score(
                labels,
                predictions,
                average="macro",
                zero_division=zero_division,
            ),
            "macro_recall": recall_score(
                labels,
                predictions,
                average="macro",
                zero_division=zero_division,
            ),
            "macro_f1": f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=zero_division,
            ),
            "weighted_precision": precision_score(
                labels,
                predictions,
                average="weighted",
                zero_division=zero_division,
            ),
            "weighted_recall": recall_score(
                labels,
                predictions,
                average="weighted",
                zero_division=zero_division,
            ),
            "weighted_f1": f1_score(
                labels,
                predictions,
                average="weighted",
                zero_division=zero_division,
            ),
        }


def resolve_paths(input_dict):
    def resolve_path(path):
        return os.path.abspath(path)

    def resolve_dict_paths(d):
        for key, value in d.items():
            if isinstance(value, str) and os.path.exists(value):
                d[key] = resolve_path(value)
            elif isinstance(value, dict):
                d[key] = resolve_dict_paths(value)
        return d

    return resolve_dict_paths(input_dict)
