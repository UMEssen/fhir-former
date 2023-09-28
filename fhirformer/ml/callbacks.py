import logging
from typing import List

import numpy as np
import torch
from datasets import load_metric
from transformers import EarlyStoppingCallback, TrainerCallback

import wandb


class BestScoreLoggingCallback(TrainerCallback):
    def __init__(self):
        self.best_scores = {}

    def on_log(self, args, state, control, model, tokenizer, logs=None, **kwargs):
        if logs is None:
            return

        def log_best(keywords: List = None) -> None:
            if keywords is None:
                keys_to_include = logs.keys()
            else:
                # Generate the keys_to_include list based on the specified keyword
                keys_to_include = [
                    k for k in logs.keys() if any(keyword in k for keyword in keywords)
                ]
            # Filter the logs dictionary to only include keys that are in the specified list
            filtered_logs = {k: v for k, v in logs.items() if k in keys_to_include}

            # Check if a new best score is achieved
            temp_dict = {}
            for key, value in filtered_logs.items():
                if key not in self.best_scores or self.best_scores[key] < value:
                    self.best_scores[key] = value
                    temp_dict[f"{key}.best"] = value

            wandb.log(temp_dict)

        # Define the keywords to include
        # log_best(["macro_f1", "weighted_f1", "macro_top10_f1", "weighted_top10_f1"])
        log_best()


class TrainingLossLoggingCallback(TrainerCallback):
    def on_train_end(self, args, state, control, logs=None, **kwargs):
        if (
            state.global_step > 0
            and state.global_step % args.logging_steps == 0
            and "loss" in state.log_history[-1]
        ):
            logs["train_loss"] = np.round(state.log_history[-1]["loss"], 4)
            wandb.log({"train_loss": logs["train_loss"]})


class DelayedEarlyStoppingCallback(TrainerCallback):
    def __init__(
        self,
        early_stopping_patience: int,
        early_stopping_threshold: int,
        delay_epochs: int,
    ):
        super().__init__()
        self.delay_epochs = delay_epochs
        self.early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        )

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.global_step > self.delay_epochs:
            logging.info("Early stopping callback kicking in...")
            self.early_stopping_callback.on_epoch_end(args, state, control, **kwargs)


class PerplexityLoggingCallback(TrainerCallback):
    def __init__(self, compute_metrics_fn):
        super().__init__()
        self.compute_metrics_fn = compute_metrics_fn

    def on_evaluate(self, args, state, control, model, metrics, **kwargs):
        epoch = state.epoch
        if epoch > 0 and epoch % 5 == 0:
            perplexity = self.compute_metrics_fn(metrics)["eval_perplexity"]
            logging.info(f"Perplexity at epoch {epoch}: {perplexity}")


class PerplexityEvaluationCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, subset_percentage=0.25):
        self.eval_dataset = eval_dataset.shuffle(seed=42)
        subset_len = int(subset_percentage * len(self.eval_dataset))
        self.eval_dataset = self.eval_dataset.select(range(subset_len))
        self.tokenizer = tokenizer
        self.perplexity_metric = load_metric("perplexity")

    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        # Perform evaluation on a subset of the validation dataset
        eval_dataloader = torch.utils.data.DataLoader(
            self.eval_dataset, batch_size=args.per_device_eval_batch_size
        )

        model.eval()
        all_logits = []
        all_labels = []
        for batch in eval_dataloader:
            with torch.no_grad():
                inputs = {k: v.to(args.device) for k, v in batch.items()}
                outputs = model(**inputs)
                logits = outputs.logits
                labels = inputs["labels"]
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)
        # Compute perplexity
        perplexity = self.perplexity_metric.compute(
            predictions=all_logits, references=all_labels
        )
        # Log perplexity
        logging.info(f"Perplexity: {perplexity}")
