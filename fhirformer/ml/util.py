import os

import numpy as np
import wandb
from sklearn.metrics import f1_score, precision_score, recall_score


def get_param_for_task_model(config, param: str, task: str, model: str):
    if task in config[param]:
        if isinstance(config[param][task], dict) and model in config[param][task]:
            return config[param][task][model]
        else:
            return config[param][task]["default"]
    return config[param]["default"]


def init_wandb(config):
    project_name = (
        "fhirformer" + "_" + "ds"
        if config["task"].startswith("ds_")
        else "fhirformer" + "_" + "pretraining"
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
    predictions: np.ndarray, labels: np.ndarray, single_label=False
):
    zero_division = 0
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
