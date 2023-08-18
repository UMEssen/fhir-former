from typing import List

import wandb
from transformers import TrainerCallback


class BestScoreLoggingCallback(TrainerCallback):
    def __init__(self):
        self.best_scores = {}

    def on_log(
            self, args, state, control, model, tokenizer, logs=None, **kwargs
    ):
        if logs is None:
            return

        def log_best(keywords: List) -> None:
            # Generate the keys_to_include list based on the specified keywords
            keys_to_include = [
                k
                for k in logs.keys()
                if any(keyword in k for keyword in keywords)
            ]

            # Filter the logs dictionary to only include keys that are in the specified list
            filtered_logs = {
                k: v for k, v in logs.items() if k in keys_to_include
            }

            # Check if a new best score is achieved
            temp_dict = {}
            for key, value in filtered_logs.items():
                prefix = key.split("_")[1]
                if key not in self.best_scores or self.best_scores[key] < value:
                    self.best_scores[key] = value
                    temp_dict[f"eval/{prefix}_f1.best"] = value
                    temp_dict[f"eval/{prefix}_precision.best"] = logs[
                        f"eval_{prefix}_precision"
                    ]
                    temp_dict[f"eval/{prefix}_recall.best"] = logs[
                        f"eval_{prefix}_recall"
                    ]
                    temp_dict[f"eval/{prefix}_accuracy.best"] = logs[
                        f"eval_accuracy"
                    ]
                    temp_dict[f"eval/{prefix}_loss.best"] = logs[f"eval_loss"]
                    temp_dict[f"eval/{prefix}_epoch.best"] = logs["epoch"]

            wandb.log(temp_dict)

        # Define the keywords to include
        log_best(
            ["macro_f1", "weighted_f1", "macro_top10_f1", "weighted_top10_f1"]
        )

class TrainingLossLoggingCallback(TrainerCallback):
    def on_train_end(self, args, state, control, logs=None, **kwargs):
        if (
            state.global_step > 0
            and state.global_step % args.logging_steps == 0
            and "loss" in state.log_history[-1]
        ):
            logs["train_loss"] = np.round(state.log_history[-1]["loss"], 4)