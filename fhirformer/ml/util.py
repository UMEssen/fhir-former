import os
from typing import Tuple

import numpy as np
import wandb
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold


# Function to determine which files to exclude
def exclude_fn(filepath):
    if any(
        excluded in filepath
        for excluded in [".venv", ".envrc", ".pre-commit-config.yaml"]
    ):
        return True
    return False


def get_param_for_task_model(config, param: str, task: str, model: str):
    if task in config[param]:
        if isinstance(config[param][task], dict) and model in config[param][task]:
            return config[param][task][model]
        else:
            return config[param][task]["default"]
    return config[param]["default"]


def split_dataset(
    dataset: Dataset, train_ratio: float = 0.8, ignore_labels: bool = False
) -> Tuple[Dataset, Dataset]:
    # TODO: Fix this to make it work with other splits, now it's just for ease of use
    assert train_ratio == 0.8, "Only 80/20 split is supported for now."
    labels = [0] * len(dataset)
    if not ignore_labels:
        if "multiclass_labels" in dataset.column_names:
            labels = dataset["multiclass_labels"]
        elif "labels" in dataset.column_names:
            labels = dataset["labels"]

    # Split the dataset into training and validation sets
    splitter = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)
    split = splitter.split(dataset, y=labels, groups=dataset["patient_id"])
    train_inds, val_inds = next(split)
    return dataset.select(train_inds), dataset.select(val_inds)


def init_wandb(config):
    project_name = (
        "fhirformer_ds_v2"
        if config["task"].startswith("ds_")
        else "fhirformer_pretraining"
    )

    wandb.init(
        tags=["baseline"],
        project=project_name,
        name=config["model_name"]
        + "_"
        + config["run_id"]
        + "_sampling_"
        + config["data_id"][config["task"]],
        mode="disabled" if config["debug"] else "online",
        entity="ship-ai-autopilot",
        group=config["task"].split("_")[1],
        resume=config["run_name"] if config["model_checkpoint"] else None,
    )

    wandb.run.log_code(
        root=".",
        include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml"),
        exclude_fn=exclude_fn,
    )


def get_evaluation_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    single_label=False,
    zero_division=0,
):
    metrics = {
        "accuracy": (predictions == labels).mean(),
    }
    if single_label:
        metrics["precision"] = precision_score(
            labels,
            predictions,
            zero_division=zero_division,
        )
        metrics["recall"] = recall_score(
            labels,
            predictions,
            zero_division=zero_division,
        )
        metrics["f1"] = f1_score(
            labels,
            predictions,
            zero_division=zero_division,
        )
    else:
        for average in ["macro", "micro", "weighted"]:
            metrics[f"{average}_precision"] = precision_score(
                labels,
                predictions,
                average=average,
                zero_division=zero_division,
            )
            metrics[f"{average}_recall"] = recall_score(
                labels,
                predictions,
                average=average,
                zero_division=zero_division,
            )
            metrics[f"{average}_f1"] = f1_score(
                labels,
                predictions,
                average=average,
                zero_division=zero_division,
            )
    return metrics


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
