import json

import evaluate
import numpy as np
import torch
import wandb
from sklearn.preprocessing import LabelBinarizer
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

os.environ["WANDB_PROJECT"] = "icd_former"
os.environ["WANDB_LOG_MODEL"] = "end"


class PatientEncounterDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=None, num_samples=None):
        with open(file_path, "r") as f:
            self.data = json.load(f)[:num_samples] if num_samples else json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

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
        label = item["label"]
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
            "label_code": item["label"],
            "label": self.label_to_id[item["label"]],
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    probabilities = softmax(logits)

    # Compute accuracy, F1-score, precision, and recall
    accuracy = evaluate.load("accuracy").compute(
        predictions=predictions, references=labels
    )
    f1 = evaluate.load("f1").compute(
        predictions=predictions, references=labels, average="weighted"
    )
    precision = precision_score(
        labels, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(labels, predictions, average="weighted", zero_division=0)

    # Compute micro and macro F1-score
    micro_f1 = f1_score(labels, predictions, average="micro")
    macro_f1 = f1_score(labels, predictions, average="macro")

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

    return {
        "accuracy": f"{accuracy['accuracy']:.2f}",
        "precision": f"{precision:.2f}",
        "recall": f"{recall:.2f}",
        "weighted_f1": f"{f1['f1']:.2f}",
        "micro_f1": f"{micro_f1:.2f}",
        "macro_f1": f"{macro_f1:.2f}",
        "auc_roc": f"{auc_roc:.2f}" if auc_roc else None,
        "macro_auc_pr": f"{macro_auc_pr:.2f}" if macro_auc_pr else None,
        "micro_auc_pr": f"{micro_auc_pr:.2f}" if micro_auc_pr else None,
    }


def train_sequence_classification_model(
    file_path: str, model_checkpoint: str, batch_size: int = 2, epochs: int = 2
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    dataset = PatientEncounterDataset(file_path, tokenizer, num_samples=20)

    # Calculate the lengths of the training and validation sets
    train_ratio = 0.8
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    val_size = dataset_size - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=len(dataset.label_to_id)
    )

    training_args = TrainingArguments(
        output_dir="/local/work/merengelke/icd_pred/results/results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        logging_dir="/local/work/merengelke/icd_pred/results/logs",
        evaluation_strategy="epoch",
        # report_to="none",
        save_total_limit=5,
        save_strategy=IntervalStrategy.EPOCH,
        fp16=True,
        weight_decay=2e-5 * 0.1,
        learning_rate=2e-5,
        load_best_model_at_end=True,
    )

    class TrainingLossLoggingCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.global_step > 0 and state.global_step % args.logging_steps == 0:
                logs["train_loss"] = np.round(state.log_history[-1]["loss"], 4)

    class BestScoreLoggingCallback(TrainerCallback):
        def __init__(self):
            self.best_scores = {}

        def on_log(self, args, state, control, model, tokenizer, logs=None, **kwargs):
            if logs is None:
                return

            # Define the keywords to include
            keywords = ["f1", "auc", "precision", "recall", "accuracy"]

            # Generate the keys_to_include list based on the specified keywords
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
                    temp_dict[f"best_{key}"] = value

            wandb.log(temp_dict)

    wandb.init(project="icd_former", name=f"{model_checkpoint}")
    wandb.run.log_code(".")

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,  # Number of steps with no improvement after which training will be stopped
        early_stopping_threshold=0.0,  # Minimum change in the monitored metric to be considered as an improvement
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            TrainingLossLoggingCallback,
            BestScoreLoggingCallback,
            early_stopping_callback,
        ],
    )

    trainer.train()

    trainer.save_model("/local/work/merengelke/icd_pred/results/model")


def main(config):
    model_checkpoint = "whaleloops/KEPTlongformer-PMM3"
    train_sequence_classification_model(config["sample_path"], model_checkpoint)


# Usage example
if __name__ == "__main__":
    main()
