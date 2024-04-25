import json
import logging
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    IntervalStrategy,
    Trainer,
    TrainingArguments,
)

import wandb
from fhirformer.helper.util import get_labels_info
from fhirformer.ml.callbacks import (
    BestScoreLoggingCallback,
    DelayedEarlyStoppingCallback,
    TrainingLossLoggingCallback,
)
from fhirformer.ml.util import get_param_for_task_model, split_dataset

logger = logging.getLogger(__name__)
os.environ["WANDB_LOG_MODEL"] = "end"

LOGGED_METRICS = [
    ("precision", "max"),
    ("recall", "max"),
    ("f1", "max"),
    ("loss", "min"),
]
METRIC_VERSIONS = ["macro", "weighted"]


class DownstreamTask:
    def __init__(
        self,
        config,
        problem_type: str,
        prediction_cutoff: float = 0.5,
    ):
        logger.info("Starting downstream task training ...")
        self.config = config
        self.model_checkpoint = config["model_checkpoint"]
        self.batch_size = config["batch_size"]
        self.epochs = config["num_train_epochs"]
        self.train_ratio = config["train_ratio"]
        self.problem_type = problem_type
        self.model, self.tokenizer = None, None

        if self.config["loaded_model"]:
            logger.info(f"Using model from checkpoint {config['model_checkpoint']}")
            wandb.run.tags = wandb.run.tags + ("pretrained",)
        self.prediction_cutoff = prediction_cutoff
        self.model_best_path = config["model_dir"] / "best"

        # Prepare dataset
        split = (
            "train"
            if self.config["max_train_samples"] is None
            else f"train[:{self.config['max_train_samples']}]"
        )
        self.dataset = load_dataset(str(config["sample_dir"]), split=split)

        self.set_up_dataset_labels()

        self.train_dataset, self.val_dataset = split_dataset(
            self.dataset, train_ratio=self.train_ratio
        )
        get_labels_info(
            labels=(
                self.train_dataset["decoded_labels"]
                if "decoded_labels" in self.train_dataset.features
                else self.train_dataset["labels"]
            ),
            additional_string="Train",
        )
        get_labels_info(
            labels=(
                self.val_dataset["decoded_labels"]
                if "decoded_labels" in self.val_dataset.features
                else self.val_dataset["labels"]
            ),
            additional_string="Validation",
        )
        logger.info(
            f"Total samples: {len(self.dataset)}, "
            f"train: {len(self.train_dataset)} "
            f"({len(self.train_dataset) / len(self.dataset) * 100:.2f}%), "
            f"validation: {len(self.val_dataset)} "
            f"({len(self.val_dataset) / len(self.dataset) * 100:.2f}%)."
        )

        weight_decay = float(
            get_param_for_task_model(
                config,
                "weight_decay",
                self.config["task"],
                self.config["model"],
            )
        )
        learning_rate = float(
            get_param_for_task_model(
                config,
                "learning_rate",
                self.config["task"],
                self.config["model"],
            )
        )
        self.training_arguments = dict(
            output_dir=self.config["model_dir"],
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            logging_dir=self.config["model_dir"] / "logs",
            evaluation_strategy="epoch",
            save_total_limit=2,
            save_strategy=IntervalStrategy.EPOCH,
            fp16=False,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )

    def set_up_models(self, num_labels: int):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_checkpoint,
            num_labels=num_labels,
            problem_type=self.problem_type,
        )
        model_path = Path(self.config["model_checkpoint"])
        if model_path.exists() and not (model_path / "tokenizer.json").exists():
            with (model_path / "config.json").open("rb") as f:
                model_name = json.load(f)["_name_or_path"]

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(str(model_path))
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_checkpoint"])

    def set_up_dataset_labels(self):
        # Doesn't do anything, the dataset stays like it was
        pass

    def tokenize_datasets(self):
        self.train_dataset = self.tokenize(self.train_dataset)
        self.val_dataset = self.tokenize(self.val_dataset)

    def tokenize(self, dataset):
        return dataset.map(
            lambda examples: self.tokenizer(
                examples["text"],
                truncation=self.config["truncation"],
                padding="max_length",
                max_length=None,
            ),
            desc="Running tokenizer on dataset",
        )

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def set_id_to_label(self, labels):
        self.model.config.id2label = {i: val for i, val in enumerate(labels)}
        self.model.config.label2id = {val: i for i, val in enumerate(labels)}

    def compute_metrics(self, eval_pred):
        raise NotImplementedError

    def train(self) -> None:
        training_args = TrainingArguments(**self.training_arguments)
        self.model_best_path.mkdir(parents=True, exist_ok=True)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[
                TrainingLossLoggingCallback,
                BestScoreLoggingCallback,
                DelayedEarlyStoppingCallback(
                    early_stopping_patience=3,  # stop training after 3 steps with no improvement
                    early_stopping_threshold=0.01,  # consider it an improvement if the metric changes by at least 0.01
                    delay_epochs=2,
                ),
            ],
        )

        trainer.train()
        trainer.save_model(self.model_best_path)
        self.tokenizer.save_pretrained(self.model_best_path)
