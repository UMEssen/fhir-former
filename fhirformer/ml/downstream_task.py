import json
import logging
import os
import warnings
from pathlib import Path

import numpy as np
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from fhirformer.helper.util import get_labels_info
from fhirformer.ml.util import get_param_for_task_model, remove_samples, split_dataset

logger = logging.getLogger(__name__)
os.environ["WANDB_LOG_MODEL"] = "end"

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings(
    "ignore",
    message="Was asked to gather along dimension 0, but all input tensors were scalars; will instead unsqueeze and return a vector.",
    module="torch.nn.parallel._functions",
)


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
        self.epochs = 5 if config["debug"] else config["num_train_epochs"]
        self.train_ratio = config["train_ratio"]
        self.problem_type = problem_type
        self.model, self.tokenizer = None, None

        if self.config["loaded_model"]:
            logger.info(f"Using model from checkpoint {config['model_checkpoint']}")
            wandb.run.tags = wandb.run.tags + ("pretrained",)
        self.prediction_cutoff = prediction_cutoff
        self.model_best_path = config["model_dir"] / "best"

        if config["is_sweep"]:
            dataset = load_dataset(
                str(config["sample_dir"]),
                split="train",
                num_proc=int(self.config["num_processes"] * 0.5),
            )
            dataset = dataset.shuffle(seed=42)
            num_samples = min(100000, len(dataset))
            self.dataset = dataset.select(range(num_samples))
        else:
            # Prepare dataset
            split = (
                "train"
                if self.config["max_train_samples"] is None
                else f"train[:{self.config['max_train_samples']}]"
            )
            self.dataset = load_dataset(
                str(config["sample_dir"]),
                split=split,
                num_proc=int(self.config["num_processes"] * 0.5),
            )

        self.test_dataset = load_dataset(str(config["sample_dir"]), split="test")
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
        get_labels_info(
            labels=(
                self.test_dataset["decoded_labels"]
                if "decoded_labels" in self.test_dataset.features
                else self.test_dataset["labels"]
            ),
            additional_string="Test",
        )
        logger.info(
            f"Total samples: {len(self.dataset)}, "
            f"train: {len(self.train_dataset)} "
            f"({len(self.train_dataset) / len(self.dataset) * 100:.2f}%), "
            f"validation: {len(self.val_dataset)} "
            f"({len(self.val_dataset) / len(self.dataset) * 100:.2f}%)."
            f"test: {len(self.test_dataset)} "
            f"({len(self.test_dataset) / len(self.dataset) * 100:.2f}%)."
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
            per_device_eval_batch_size=self.batch_size,
            logging_dir=self.config["model_dir"] / "logs",
            eval_strategy="steps",  # eval_strategy
            eval_steps=0.01,  # adapt depending on dataset size / epochs
            logging_strategy="steps",
            logging_steps=0.005,
            save_total_limit=2,
            save_strategy="steps",
            save_steps=0.01,
            fp16=False,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_steps=500,
        )

    def set_up_models(self, num_labels: int):
        logging.info(f"Setting up model with {num_labels} labels")
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_checkpoint"], weights_only=True
        )

    def set_up_dataset_labels(self):
        print("Setting up dataset labels ds task")
        # Doesn't do anything, the dataset stays like it was
        pass

    def tokenize_datasets(self):
        self.train_dataset = self.tokenize(self.train_dataset)
        self.val_dataset = self.tokenize(self.val_dataset)
        self.test_dataset = self.tokenize(self.test_dataset)

    def tokenize(self, dataset):
        return dataset.map(
            lambda examples: self.tokenizer(
                examples["text"],
                truncation=self.config["truncation"],
                padding="max_length",
                max_length=None,
            ),
            desc="Running tokenizer on dataset",
            num_proc=int(self.config["num_processes"] * 0.5),
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

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=5,
                    early_stopping_threshold=0.01,
                )
            ],
        )

        self.trainer.train()
        self.trainer.save_model(self.model_best_path)
        self.tokenizer.save_pretrained(self.model_best_path)

        self.test()

        if self.config["is_sweep"]:
            remove_samples(self.config)

    def test(self):
        logger.info("Evaluating the model on the test dataset...")

        test_results = self.trainer.evaluate(
            eval_dataset=self.test_dataset, metric_key_prefix="test"
        )
        for key, value in test_results.items():
            logger.info(f"{key}: {value:.4f}")
        wandb.log(test_results)
