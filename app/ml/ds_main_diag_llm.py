import json
import datetime
import logging
from pathlib import Path

import numpy as np
import torch
import wandb
from sklearn.preprocessing import LabelBinarizer
from torch.nn import BCELoss
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    IntervalStrategy,
    EarlyStoppingCallback,
)
import os
from torch.utils.data import random_split
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from transformers import TrainerCallback

from typing import Dict, Union

from app.ml import ds_icd_llm, train_helper

os.environ["WANDB_LOG_MODEL"] = "end"


class PatientEncounterDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=None, num_samples=None):
        with open(file_path, "r") as f:
            self.data = json.load(f)[:num_samples] if num_samples else json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lb = LabelBinarizer()
        self.labels = self.lb.fit_transform([item["label"] for item in self.data])

        icds = [item["label"] for item in self.data]
        # Create a mapping of unique root ICD-10 codes to integers
        self.label_to_id = {}
        for icd in icds:
            if icd not in self.label_to_id:
                self.label_to_id[icd] = len(self.label_to_id)

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
            "label_codes": self.lb.classes_,
            "label_code": item["label"],
            "label": self.label_to_id[item["label"]],
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "decoded_labels": [
                self.lb.classes_[i]
                for i, label in enumerate(self.labels[idx])
                if label == 1
            ],
        }


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class DS_ICD_Main:
    def __init__(
        self,
        config,
        model_checkpoint: str,
        batch_size: int = 2,
        epochs: int = 2,
        sample_size: int = None,
    ):
        self.config = config
        run = wandb.init(
            project="icd_main_former",
            name=f"{model_checkpoint}",
            mode="online",
            # mode="disabled",
            tags=["demo"],
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

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.dataset = PatientEncounterDataset(
            "/local/work/merengelke/main_diagnosis_classifier/icd_pred/samples.json",
            tokenizer,
            num_samples=sample_size,
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
            self.dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        logging.info(
            f"total samples: {len(self.train_dataset) + len(self.val_dataset)}"
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_checkpoint, num_labels=len(self.dataset.label_to_id)
        )

        self.model.config.problem_type = (
            "single_label_classification"  # specify problem type
        )

        self.model.loss = BCELoss()  # single class classification

        labels = [label for label in self.train_dataset[0]["label_codes"]]

        self.model.config.id2label = {i: val for i, val in enumerate(labels)}
        self.model.config.label2id = {val: i for i, val in enumerate(labels)}

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        probabilities = softmax(logits)

        # Compute accuracy, F1-score, precision, and recall
        accuracy = (predictions == labels).mean()

        # Compute sample-based precision, recall, and F1-score
        macro_precision = precision_score(labels, predictions, average="macro")
        macro_recall = recall_score(labels, predictions, average="macro")
        macro_f1 = f1_score(labels, predictions, average="macro")

        weighted_precision = precision_score(labels, predictions, average="weighted")
        weighted_recall = recall_score(labels, predictions, average="weighted")
        weighted_f1 = f1_score(labels, predictions, average="weighted")

        # Create a binary representation of the labels and probabilities
        unique_classes = np.unique(np.concatenate((labels, predictions)))
        lb = LabelBinarizer()
        lb.fit(unique_classes)
        binary_labels = lb.transform(labels)
        binary_probabilities = lb.transform(predictions)

        # Compute AUC-ROC and AUC-PR only if there are at least two unique classes
        # if len(unique_classes) > 1:
        try:
            auc_roc = roc_auc_score(
                binary_labels, probabilities[:, unique_classes], multi_class="ovr"
            )
            macro_auc_pr = average_precision_score(
                binary_labels, probabilities[:, unique_classes], average="macro"
            )
            micro_auc_pr = average_precision_score(
                binary_labels, probabilities[:, unique_classes], average="micro"
            )
        except ValueError:
            # Assign default values or skip these metrics
            auc_roc = None
            macro_auc_pr = None
            micro_auc_pr = None

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

        # Extract the top 10 labels and predictions
        mask = np.isin(predictions, self.top10_classes)
        labels_top10 = labels[mask]
        predictions_top10 = predictions[mask]

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
            "auc_roc": f"{auc_roc:.2f}" if auc_roc else None,
            "macro_auc_pr": f"{macro_auc_pr:.2f}" if macro_auc_pr else None,
            "micro_auc_pr": f"{micro_auc_pr:.2f}" if micro_auc_pr else None,
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

        print("Path logs: ", self.config["model_dir"] / Path("logs"))

        # Load the dataset
        training_args = TrainingArguments(
            output_dir=self.config["model_dir"],
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            logging_dir=self.config["model_dir"] / Path("logs"),
            evaluation_strategy="epoch",
            # report_to="none",
            save_total_limit=3,
            save_strategy=IntervalStrategy.EPOCH,
            fp16=True,
            weight_decay=2e-5 * 0.1,
            learning_rate=2e-5,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False
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
        self.model.config.to_json_file(
            str(self.config["model_dir"] / "models" / "config.json")
        )


def main(config):
    model_checkpoint = "whaleloops/KEPTlongformer-PMM3"
    config["model_dir"] = config["model_dir"] / Path(
        model_checkpoint.replace("/", "_")
        + "_"
        + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    )
    config["model_dir"].mkdir(parents=True, exist_ok=True)

    ds_icd_main = DS_ICD_Main(config, model_checkpoint, epochs=50, sample_size=None)
    ds_icd_main.train_sequence_classification_model()
