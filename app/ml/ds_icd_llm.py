import datetime
import json
import logging
import os
from pathlib import Path

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
from torch.utils.data import random_split
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from transformers import TrainerCallback
from typing import Dict, Union, List

from app.ml import train_helper


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
            "decoded_labels": [
                self.mlb.classes_[i]
                for i, label in enumerate(self.labels[idx])
                if label == 1
            ],
        }


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class DS_Task_ICD_Predict:
    def __init__(
        self, config, model_checkpoint: str, batch_size: int = 2, epochs: int = 2
    ):
        self.config = config
        run = wandb.init(
            project="ds_icd_former",
            name=f"{model_checkpoint}",
            mode="online",
            # mode="disabled",
            tags=["test"],
        )
        self.model_checkpoint = (
            model_checkpoint
            if not config["artifact"]
            else run.use_artifact(config["artifact"], type="model").download()
        )
        self.batch_size = batch_size
        self.epochs = epochs

        if self.config["artifact"]:
            logging.info("Using artifact")
            logging.info(
                "Make sure the model_checkpoint matches the artifact base model"
            )
            print(self.model_checkpoint)
            print(os.listdir(self.model_checkpoint))
            print(type(self.model_checkpoint))
            run.tags = run.tags + ("pretrained",)

        # todo load the tokenizer from the artifact once it is avaiable
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.dataset = PatientEncounterDataset(
            self.config["ds_icd_train_sample_path"], tokenizer, num_samples=None
        )
        label_freq = np.sum(
            self.dataset.labels, axis=0
        )  # sum over the column (each label)
        self.top10_classes = label_freq.argsort()[-10:][
            ::-1
        ]  # find top 10 most common classes

        # Calculate the lengths of the training and validation sets
        train_ratio = 0.8
        dataset_size = len(self.dataset)
        train_size = int(dataset_size * train_ratio)
        val_size = dataset_size - train_size

        # Split the dataset into training and validation sets
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size]
        )
        logging.info(f"total samples: {len(self.train_dataset)+len(self.val_dataset)}")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_checkpoint, num_labels=len(self.dataset.mlb.classes_)
        )

        self.model.config.problem_type = (
            "multi_label_classification"  # specify problem type
        )
        self.model.loss = BCEWithLogitsLoss()  # specify loss function for multi-label

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

        macro_top10_precision = precision_score(
            labels_top10, predictions_top10, average="macro"
        )
        macro_top10_recall = recall_score(
            labels_top10, predictions_top10, average="macro"
        )
        macro_top10_f1 = f1_score(labels_top10, predictions_top10, average="macro")
        weighted_top10_precision = precision_score(
            labels_top10, predictions_top10, average="weighted"
        )
        weighted_top10_recall = recall_score(
            labels_top10, predictions_top10, average="weighted"
        )
        weighted_top10_f1 = f1_score(
            labels_top10, predictions_top10, average="weighted"
        )

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
            "weighted_top10_f1": np.round(weighted_top10_f1, 2),
        }

    def train_sequence_classification_model(self) -> None:
        wandb.run.log_code(".")
        wandb.define_metric("eval/macro_f1", summary="max", step_metric="epoch")
        wandb.define_metric("eval/accuracy", summary="max", step_metric="epoch")
        wandb.define_metric("eval/macro_loss", summary="min", step_metric="epoch")
        wandb.define_metric("eval/macro_precision", summary="max", step_metric="epoch")
        wandb.define_metric("eval/macro_recall", summary="max", step_metric="epoch")
        wandb.define_metric("eval/weighted_f1", summary="max", step_metric="epoch")
        wandb.define_metric("eval/weighted_loss", summary="min", step_metric="epoch")
        wandb.define_metric(
            "eval/weighted_precision", summary="max", step_metric="epoch"
        )
        wandb.define_metric("eval/weighted_recall", summary="max", step_metric="epoch")
        wandb.define_metric(
            "eval/macro_top10_precision", summary="max", step_metric="epoch"
        )
        wandb.define_metric(
            "eval/macro_top10_recall", summary="max", step_metric="epoch"
        )
        wandb.define_metric("eval/macro_top10_f1", summary="max", step_metric="epoch")
        wandb.define_metric(
            "eval/weighted_top10_precision", summary="max", step_metric="epoch"
        )
        wandb.define_metric(
            "eval/weighted_top10_recall", summary="max", step_metric="epoch"
        )
        wandb.define_metric(
            "eval/weighted_top10_f1", summary="max", step_metric="epoch"
        )

        training_args = TrainingArguments(
            output_dir=self.config["model_dir"],
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            logging_dir=self.config["model_dir"] / Path("logs"),
            evaluation_strategy="epoch",
            save_total_limit=2,
            save_strategy=IntervalStrategy.EPOCH,
            fp16=True,
            weight_decay=2e-5 * 0.1,
            learning_rate=2e-5,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
        )




        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=5,  # Number of steps with no improvement after which training will be stopped
            early_stopping_threshold=0.0,  # Minimum change in the monitored metric to be considered as an improvement
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[
                train_helper.TrainingLossLoggingCallback,
                train_helper.BestScoreLoggingCallback,
                early_stopping_callback,
            ],
        )

        trainer.train()

        trainer.save_model(self.config["model_dir"] / Path("models"))

        config = trainer.model.config
        labels = [label for label in self.train_dataset[0]["label_codes"]]
        config.id2label = {i: val for i, val in enumerate(labels)}
        config.label2id = {val: i for i, val in enumerate(labels)}
        config.to_json_file(str(self.config["model_dir"] / "models" / "config.json"))


def main(config):
    model_checkpoint = "LennartKeller/longformer-gottbert-base-8192-aw512"
    config["model_dir"] = config["model_dir"] / Path(
        model_checkpoint.replace("/", "_")
        + "_"
        + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    )
    config["model_dir"].mkdir(parents=True, exist_ok=True)

    ds_icd = DS_Task_ICD_Predict(config, model_checkpoint, epochs=10)
    ds_icd.train_sequence_classification_model()


# Usage example
if __name__ == "__main__":
    main()
