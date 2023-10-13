import os

import wandb


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
    )
    wandb.run.log_code(".")


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
