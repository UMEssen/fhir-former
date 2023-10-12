import os

import wandb


def init_wandb(config):
    wandb.init(
        tags=["baseline"],
        project=config["task"],
        name=config["model_name"] + "_" + config["run_id"],
        mode="disabled" if config["debug"] else "online",
        entity="ship-ai-autopilot",
        group="DDP",
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
