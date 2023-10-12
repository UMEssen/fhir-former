import logging
import os

import numpy as np
import wandb
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import random_split
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    IntervalStrategy,
    Trainer,
    TrainingArguments,
)

from fhirformer.ml.callbacks import (
    BestScoreLoggingCallback,
    TrainingLossLoggingCallback,
)
from fhirformer.ml.util import init_wandb

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
        dataset_class,
        dataset_args,
        model_checkpoint: str,
        batch_size: int = 2,
        epochs: int = 3,
        train_ratio: float = 0.8,
        prediction_cutoff: float = 0.5,
    ):
        logger.info("Starting downstream task training ...")
        self.config = config
        self.model_checkpoint = model_checkpoint
        self.batch_size = batch_size
        self.epochs = epochs

        if self.config["load_from_file"]:
            logger.info(f"Using model from checkpoint {config['model_checkpoint']}")
            wandb.run.tags = wandb.run.tags + ("pretrained",)
        self.prediction_cutoff = prediction_cutoff
        self.model_best_path = config["model_dir"] / "best"

        # Prepare dataset
        self.dataset = dataset_class(**dataset_args)

        # Calculate the lengths of the training and validation sets
        dataset_size = len(self.dataset)
        train_size = int(dataset_size * train_ratio)
        val_size = dataset_size - train_size

        # Split the dataset into training and validation sets
        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size]
        )
        logger.info(f"Total samples: {len(self.train_dataset)+len(self.val_dataset)}")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_checkpoint,
            num_labels=self.dataset.num_classes,
            problem_type=self.dataset.problem_type,
        )
        self.training_arguments = dict(
            output_dir=self.config["model_dir"],
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            logging_dir=self.config["model_dir"] / "logs",
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

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def set_id_to_label(self, labels):
        self.model.config.id2label = {i: val for i, val in enumerate(labels)}
        self.model.config.label2id = {val: i for i, val in enumerate(labels)}

    @staticmethod
    def metrics(predictions: np.ndarray, labels: np.ndarray):
        zero_division = np.nan
        return {
            "accuracy": (predictions == labels).mean(),
            "macro_precision": precision_score(
                labels, predictions, average="macro", zero_division=zero_division
            ),
            "macro_recall": recall_score(
                labels, predictions, average="macro", zero_division=zero_division
            ),
            "macro_f1": f1_score(
                labels, predictions, average="macro", zero_division=zero_division
            ),
            "weighted_precision": precision_score(
                labels, predictions, average="weighted", zero_division=zero_division
            ),
            "weighted_recall": recall_score(
                labels, predictions, average="weighted", zero_division=zero_division
            ),
            "weighted_f1": f1_score(
                labels, predictions, average="weighted", zero_division=zero_division
            ),
        }

    def compute_metrics(self, eval_pred):
        raise NotImplementedError

    def train(self) -> None:
        training_args = TrainingArguments(**self.training_arguments)
        self.model_best_path.mkdir(parents=True, exist_ok=True)
        with training_args.main_process_first(desc="Create Dataset"):
            init_wandb(self.config)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[
                TrainingLossLoggingCallback,
                BestScoreLoggingCallback,
                EarlyStoppingCallback(
                    early_stopping_patience=5,
                    # Number of steps with no improvement after which training will be stopped
                    early_stopping_threshold=0.0,
                    # Minimum change in the monitored metric to be considered as an improvement
                ),
            ],
        )

        trainer.train()
        trainer.save_model(self.model_best_path)
