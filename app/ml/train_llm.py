import json

import evaluate
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import os
from torch.utils.data import random_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score


from typing import Dict, Union


class PatientEncounterDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=None):
        with open(file_path, "r") as f:
            self.data = json.load(f)
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
            # truncation=True,
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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = evaluate.load("accuracy").compute(
        predictions=predictions, references=labels
    )
    f1 = evaluate.load("f1").compute(
        predictions=predictions, references=labels, average="weighted"
    )

    print(labels)
    print(predictions)

    # Compute confusion matrix, tp, fp, fn, tn
    cm = confusion_matrix(labels, predictions)
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tn = cm[0, 0]

    # Compute precision and recall with zero_division parameter
    precision = precision_score(
        labels, predictions, average="weighted", zero_division=0
    )
    recall = recall_score(labels, predictions, average="weighted", zero_division=0)

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
    }


def train_sequence_classification_model(
    file_path: str, model_checkpoint: str, batch_size: int = 8, epochs: int = 3
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    dataset = PatientEncounterDataset(file_path, tokenizer)

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
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        logging_dir="./logs",
        evaluation_strategy="epoch",
        # report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()


def main(config):
    model_checkpoint = "allenai/longformer-base-4096"
    train_sequence_classification_model(config["sample_path"], model_checkpoint)


# Usage example
if __name__ == "__main__":
    main()
