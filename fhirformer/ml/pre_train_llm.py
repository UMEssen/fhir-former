import logging
import os

import numpy as np
import torch
from datasets import load_dataset
from torch.nn import functional as F
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EvalPrediction,
    IntervalStrategy,
    Trainer,
    TrainingArguments,
)

from fhirformer.ml.callbacks import TrainingLossLoggingCallback
from fhirformer.ml.util import get_param_for_task_model

logger = logging.getLogger(__name__)
os.environ["WANDB_LOG_MODEL"] = "end"


class Pretrainer:
    def __init__(self, config):
        self.model_name = config["model_checkpoint"]
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config["model_checkpoint"],
            do_lower_case=False,
            device_map="auto",
        )
        self.model_best_path = config["model_dir"] / "best"

    def compute_metrics(self, eval_pred: EvalPrediction):
        batch_size = 1 * self.config["eval_accumulation_steps"]
        logits, labels = eval_pred
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_batches = int(np.ceil(logits.shape[0] / batch_size))
        perplexity = torch.tensor(0.0).to(device)

        # Convert numpy arrays to tensors and move them to the GPU
        logits_tensor = torch.from_numpy(logits).to(device)
        labels_tensor = torch.from_numpy(labels).to(device)

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            # Take a batch from the tensors
            logits_batch = logits_tensor[start_idx:end_idx]
            labels_batch = labels_tensor[start_idx:end_idx]

            # Now calculate the loss
            masked_lm_loss = F.cross_entropy(
                logits_batch.view(-1, logits_batch.size(-1)),
                labels_batch.view(-1),
                reduction="none",
            )

            # Update perplexity
            perplexity += torch.exp(masked_lm_loss).mean()

        perplexity /= num_batches
        return {"perplexity": perplexity.item()}

    def tokenize(self, dataset):
        # (self.config["task_dir"] / "map_cache").mkdir(parents=True, exist_ok=True)
        return dataset.map(
            lambda examples: self.tokenizer(
                examples["text"],
                truncation=self.config["truncation"],
                padding="max_length",
                max_length=None,  # , return_tensors="pt"
            ),
            # cache_file_name=str(
            #     self.config["task_dir"]
            #     / "map_cache"
            #     / f"{self.config['task']}_tokenized_{dataset.split}.arrow"
            # ),
        )

    @staticmethod
    def unite_datasets(fhir_dataset, doc_dataset):
        if fhir_dataset is not None and doc_dataset is not None:
            return fhir_dataset.concatenate(doc_dataset)
        elif fhir_dataset is not None:
            return fhir_dataset
        elif doc_dataset is not None:
            return doc_dataset
        else:
            raise ValueError("No dataset found")

    def build_datasets(self, disable_validation: bool = False):
        fhir_train_dataset, fhir_val_dataset, doc_train_dataset, doc_val_dataset = (
            None,
            None,
            None,
            None,
        )
        if disable_validation:
            logger.info(
                "The validation is disabled and no validation set will be computed."
            )
        split_train = (
            "train"
            if self.config["max_train_samples"] is None
            else f"train[:{self.config['max_train_samples']}]"
        )
        split_validation = (
            "validation"
            if self.config["max_test_samples"] is None
            else f"validation[:{self.config['max_test_samples']}]"
        )
        if "_fhir" in self.config["task"]:
            fhir_train_dataset = load_dataset(
                str(self.config["sample_dir"]), split=split_train
            )
            if not disable_validation:
                fhir_val_dataset = load_dataset(
                    str(self.config["sample_dir"]), split=split_validation
                )
        if "_documents" in self.config["task"]:
            doc_train_dataset = load_dataset(
                str(self.config["task_dir"] / "sentences_deduplicated"),
                split=split_train,
            )
            if not disable_validation:
                doc_val_dataset = load_dataset(
                    str(self.config["task_dir"] / "sentences"), split=split_validation
                )

        train_dataset = self.unite_datasets(fhir_train_dataset, doc_train_dataset)
        val_dataset = None
        if not disable_validation:
            val_dataset = self.unite_datasets(fhir_val_dataset, doc_val_dataset)

        assert (
            len(train_dataset) > 0
        ), "Something went wrong with the generation of the samples."
        sum_val = len(train_dataset) + (len(val_dataset) if val_dataset else 0)
        logger.info(f"Total samples: {sum_val}")

        return (
            self.tokenize(train_dataset),
            self.tokenize(val_dataset) if val_dataset else None,
        )

    def pretrain(self):
        weight_decay = float(
            get_param_for_task_model(
                self.config, "weight_decay", self.config["task"], self.config["model"]
            )
        )
        learning_rate = float(
            get_param_for_task_model(
                self.config, "learning_rate", self.config["task"], self.config["model"]
            )
        )
        training_args = TrainingArguments(
            output_dir=self.config["model_dir"],
            overwrite_output_dir=True,
            num_train_epochs=self.config["num_train_epochs"],
            per_device_train_batch_size=self.config["batch_size"],  # per device
            save_total_limit=2,
            report_to="wandb",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            save_strategy=IntervalStrategy.EPOCH,
            fp16=False,
            # These values below are all needed for early stopping, loading the best model at end
            # and evaluation
            #
            # eval_accumulation_steps=self.config["eval_accumulation_steps"],  # tune
            # evaluation_strategy="epoch",
            # load_best_model_at_end=True,
            # per_device_eval_batch_size=1,  # per device
            # metric_for_best_model="loss",
            # greater_is_better=False,
        )

        train_dataset, val_dataset = self.build_datasets(disable_validation=True)

        logger.info("Starting pre-training...")
        self.model_best_path.mkdir(parents=True, exist_ok=True)

        # Define data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )
        model = AutoModelForMaskedLM.from_pretrained(self.model_name)

        # Create the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            # eval_dataset=val_dataset,
            # compute_metrics=self.compute_metrics,
            callbacks=[
                TrainingLossLoggingCallback,
                # EarlyStoppingCallback(
                #     early_stopping_patience=20,
                #     # Number of steps with no improvement after which training will be stopped
                #     early_stopping_threshold=0.0,
                #     # Minimum change in the monitored metric to be considered as an improvement
                # ),
            ],
        )
        torch.cuda.empty_cache()
        trainer.train()
        # Model best is currently not the best because we only save the last model
        trainer.save_model(self.model_best_path)
        self.tokenizer.save_pretrained(self.model_best_path)


def main(config):
    pretrainer = Pretrainer(config)
    # Usage example
    pretrainer.pretrain()
