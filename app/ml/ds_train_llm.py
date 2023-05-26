import json
import logging
from pathlib import Path

import evaluate
import numpy as np
import torch
import wandb
from scipy.special import expit
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    IntervalStrategy,
    EarlyStoppingCallback,
)
from torch.nn import BCEWithLogitsLoss
import os
from torch.utils.data import random_split
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from transformers import TrainerCallback

from typing import Dict, Union, List


class PatientEncounterDataset(Dataset):
    def __init__(self, train_sample_path, tokenizer, max_length=None, num_samples=None):
        with open(train_sample_path, "r") as f:
            self.data = json.load(f)[:num_samples] if num_samples else json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

        icds = [item["label"] for item in self.data]
        # Create a mapping of unique root ICD-10 codes to integers
        self.label_to_id = {}
        self.mlb = MultiLabelBinarizer()
        self.labels = self.mlb.fit_transform(icds)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, Union[int, str, torch.Tensor]]:
        item = self.data[idx]
        text = item["text"]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            # return_tensors="pt",
        )

        return {
            "patient_id": item["patient_id"],
            "encounter_id": item["encounter_id"],
            "text": item["text"],
            "label_codes": self.mlb.classes_,  # changed from "label_code" to "label_codes"
            "labels": torch.Tensor(
                self.labels[idx]
            ),  # changed from "label" to "labels"
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)





class DS_Task_ICD_Predict:
    def __init__(
        self, config, model_checkpoint: str, batch_size: int = 2, epochs: int = 2
    ):
        self.config = config
        self.model_checkpoint = model_checkpoint
        self.batch_size = batch_size
        self.epochs = epochs
        tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.dataset = PatientEncounterDataset(
            self.config["ds_icd_train_sample_path"], tokenizer, num_samples=None
        )
        label_freq = np.sum(self.dataset.labels, axis=0) # sum over the column (each label)
        self.top10_classes = label_freq.argsort()[-10:][::-1] # find top 10 most common classes

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        probabilities = expit(logits)  # sigmoid function
        predictions = (probabilities > 0.5).astype(
            int
        )  # threshold at 0.5logits, labels = eval_pred

        # Compute label-based accuracy
        accuracy = (predictions == labels).mean()

        # Compute sample-based precision, recall, and F1-score
        macro_precision = precision_score(labels, predictions, average="macro")
        macro_recall = recall_score(labels, predictions, average="macro")
        macro_f1 = f1_score(labels, predictions, average="macro")

        weighted_precision = precision_score(labels, predictions, average="weighted")
        weighted_recall = recall_score(labels, predictions, average="weighted")
        weighted_f1 = f1_score(labels, predictions, average="weighted")

        # This is a workaround to be able to use wandb.define_metric to log max and best scores of a run
        metrics = {
            "accuracy": np.round(accuracy, 2),
            "eval_macro_precision": np.round(macro_precision, 2),
            "eval_macro_recall": np.round(macro_recall, 2),
            "eval_macro_f1": np.round(macro_f1, 2),
            "eval_weighted_precision": np.round(weighted_precision, 2),
            "eval_weighted_recall": np.round(weighted_recall, 2),
            "eval_weighted_f1": np.round(weighted_f1, 2),
        }

        # Compute metrics for top 10 classes
        labels_top10 = labels[:, self.top10_classes]
        predictions_top10 = predictions[:, self.top10_classes]

        macro_top10_precision = precision_score(labels_top10, predictions_top10, average='macro')
        macro_top10_recall = recall_score(labels_top10, predictions_top10, average='macro')
        macro_top10_f1 = f1_score(labels_top10, predictions_top10, average='macro')
        weighted_top10_precision = precision_score(labels_top10, predictions_top10, average='weighted')
        weighted_top10_recall = recall_score(labels_top10, predictions_top10, average='weighted')
        weighted_top10_f1 = f1_score(labels_top10, predictions_top10, average='weighted')

        metrics["eval_macro_top10_precision"] = np.round(macro_top10_precision, 2)
        metrics["eval_macro_top10_recall"] = np.round(macro_top10_recall, 2)
        metrics["eval_macro_top10_f1"] = np.round(macro_top10_f1, 2)
        metrics["eval_weighted_top10_precision"] = np.round(weighted_top10_precision, 2)
        metrics["eval_weighted_top10_recall"] = np.round(weighted_top10_recall, 2)
        metrics["eval_weighted_top10_f1"] = np.round(weighted_top10_f1, 2)

        # Log metrics to wandb
        wandb.log(metrics)

        return {
            "accuracy": np.round(accuracy, 2),
            "macro_precision": np.round(macro_precision, 2),
            "macro_recall": np.round(macro_recall, 2),
            "macro_f1": np.round(macro_f1, 2),
            "weighted_precision": np.round(weighted_precision, 2),
            "weighted_recall": np.round(weighted_recall, 2),
            "weighted_f1": np.round(weighted_f1, 2),
            "macro_top10_precision": np.round(macro_top10_precision, 2),
            "macro_top10_recall": np.round(macro_top10_recall, 2),
            "macro_top10_f1": np.round(macro_top10_f1, 2),
            "weighted_top10_precision": np.round(weighted_top10_precision, 2),
            "weighted_top10_recall": np.round(weighted_top10_recall, 2),
            "weighted_top10_f1": np.round(weighted_top10_f1, 2)
        }

    def train_sequence_classification_model(self) -> None:
        # Calculate the lengths of the training and validation sets
        train_ratio = 0.8
        dataset_size = len(self.dataset)
        train_size = int(dataset_size * train_ratio)
        val_size = dataset_size - train_size

        # Split the dataset into training and validation sets
        train_dataset, val_dataset = random_split(self.dataset, [train_size, val_size])
        logging.info(f"total samples: {len(train_dataset)+len(val_dataset)}")

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_checkpoint, num_labels=len(self.dataset.mlb.classes_)
        )

        model.config.problem_type = "multi_label_classification"  # specify problem type
        model.loss = BCEWithLogitsLoss()  # specify loss function for multi-label

        training_args = TrainingArguments(
            output_dir=self.config["logging_dir"],
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            logging_dir=self.config["logging_dir"] / Path("logs"),
            evaluation_strategy="epoch",
            save_total_limit=2,
            save_strategy=IntervalStrategy.EPOCH,
            fp16=True,
            weight_decay=2e-5 * 0.1,
            learning_rate=2e-5,
            load_best_model_at_end=True,
        )

        class TrainingLossLoggingCallback(TrainerCallback):
            def on_train_end(self, args, state, control, logs=None, **kwargs):
                if (
                    state.global_step > 0
                    and state.global_step % args.logging_steps == 0
                    and "loss" in state.log_history[-1]
                ):
                    logs["train_loss"] = np.round(state.log_history[-1]["loss"], 4)

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
                        k for k in logs.keys() if any(keyword in k for keyword in keywords)
                    ]

                    # Filter the logs dictionary to only include keys that are in the specified list
                    filtered_logs = {k: v for k, v in logs.items() if k in keys_to_include}

                    # Check if a new best score is achieved
                    temp_dict = {}
                    for key, value in filtered_logs.items():
                        prefix = key.split("_")[1]
                        if key not in self.best_scores or self.best_scores[key] < value:
                            self.best_scores[key] = value
                            temp_dict[f"eval/{prefix}_f1.best"] = value
                            temp_dict[f"eval/{prefix}_precision.best"] = logs[f'eval_{prefix}_precision']
                            temp_dict[f"eval/{prefix}_recall.best"] = logs[f'eval_{prefix}_recall']
                            temp_dict[f"eval/{prefix}_accuracy.best"] = logs[f'eval_accuracy']
                            temp_dict[f"eval/{prefix}_loss.best"] = logs[f'eval_loss']
                            temp_dict[f"eval/{prefix}_epoch.best"] = logs['epoch']

                    wandb.log(temp_dict)

                # Define the keywords to include
                log_best(["macro_f1", "weighted_f1", "macro_top10_f1", "weighted_top10_f1"])


        wandb.init(
            project="icd_former",
            name=f"{self.model_checkpoint}",
            mode="online",
            tags=["baseline"],
        )
        wandb.run.log_code(".")
        wandb.define_metric("eval/macro_f1", summary="max", step_metric="epoch")
        wandb.define_metric("eval/accuracy", summary="max", step_metric="epoch")
        wandb.define_metric("eval/macro_loss", summary="min", step_metric="epoch")
        wandb.define_metric("eval/macro_precision", summary="max", step_metric="epoch")
        wandb.define_metric("eval/macro_recall", summary="max", step_metric="epoch")
        wandb.define_metric("eval/weighted_f1", summary="max", step_metric="epoch")
        wandb.define_metric("eval/weighted_loss", summary="min", step_metric="epoch")
        wandb.define_metric("eval/weighted_precision", summary="max", step_metric="epoch")
        wandb.define_metric("eval/weighted_recall", summary="max", step_metric="epoch")
        wandb.define_metric("eval/macro_top10_precision", summary="max", step_metric="epoch")
        wandb.define_metric("eval/macro_top10_recall", summary="max", step_metric="epoch")
        wandb.define_metric("eval/macro_top10_f1", summary="max", step_metric="epoch")
        wandb.define_metric("eval/weighted_top10_precision", summary="max", step_metric="epoch")
        wandb.define_metric("eval/weighted_top10_recall", summary="max", step_metric="epoch")
        wandb.define_metric("eval/weighted_top10_f1", summary="max", step_metric="epoch")

        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=3,  # Number of steps with no improvement after which training will be stopped
            early_stopping_threshold=0.0,  # Minimum change in the monitored metric to be considered as an improvement
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[
                TrainingLossLoggingCallback,
                BestScoreLoggingCallback,
                early_stopping_callback,
            ],
        )

        trainer.train()

        trainer.save_model(self.config["model_dir"] / Path("models"))


def main(config):
    model_checkpoint = "bert-base-uncased"
    ds_icd = DS_Task_ICD_Predict(config, model_checkpoint, epochs=150)
    ds_icd.train_sequence_classification_model()


# Usage example
if __name__ == "__main__":
    main()
