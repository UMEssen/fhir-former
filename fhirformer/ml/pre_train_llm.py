import json
import logging
import os
from functools import partial

import numpy as np
import torch
from datasets import Dataset as HFDataset
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

logger = logging.getLogger(__name__)
os.environ["WANDB_LOG_MODEL"] = "end"


class Pretrainer:
    def __init__(self, config, tokenizer=None):
        self.model_name = config["model_checkpoint"]
        self.config = config
        self.tokenizer = (
            tokenizer
            if tokenizer
            else AutoTokenizer.from_pretrained(config["model_checkpoint"])
        )
        self.model_best_path = config["model_dir"] / "best"
        self.model_best_path.mkdir(parents=True, exist_ok=True)

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

    def get_text(self, phase: str):
        num_samples = None
        if self.config["debug"]:
            num_samples = 100
        with open(self.config["task_dir"] / f"{phase}.json", "r") as f:
            train_texts = json.load(f)[:num_samples] if num_samples else json.load(f)
        return train_texts

    def data_generator(self, phase: str):
        for text in self.get_text(phase):
            yield {"text": text["text"]}

    def tokenize(self, dataset):
        return dataset.map(
            lambda examples: self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=None,  # , return_tensors="pt"
            )
        )

    def get_samples_dataset(self, phase: str):
        return HFDataset.from_generator(
            partial(self.data_generator, phase),
            cache_dir=self.config["task_dir"] / "hf_cache",
        )

    def get_document_dataset(self):
        from fhirformer.data_preprocessing.document_dataset import DocumentDataset

        ds = DocumentDataset(
            config_name=self.config["task"],
            data_dir=self.config["task_dir"] / "doc_data",
            document_folder=self.config["data_dir"] / "documents",
            task_folder=self.config["task_dir"],
        )
        ds.download_and_prepare(output_dir=self.config["task_dir"] / "doc_data")
        data = ds.as_dataset()
        train_dataset = data["train"]
        test_dataset = data["test"]
        logger.info(f"train {len(train_dataset)}, test {len(test_dataset)}")
        return train_dataset, test_dataset

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

    def pretrain(self, num_train_epochs=2):
        training_args = TrainingArguments(
            output_dir=self.config["model_dir"],
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=torch.cuda.device_count(),  # per device
            per_device_eval_batch_size=1,  # per device
            save_total_limit=2,
            load_best_model_at_end=True,
            eval_accumulation_steps=self.config["eval_accumulation_steps"],  # tune
            report_to="wandb",
            evaluation_strategy="epoch",
            save_strategy=IntervalStrategy.EPOCH,
            fp16=True,
            metric_for_best_model="loss",
            greater_is_better=False,
        )

        with training_args.main_process_first(desc="Create Dataset"):
            fhir_train_dataset, fhir_val_dataset, doc_train_dataset, doc_val_dataset = (
                None,
                None,
                None,
                None,
            )
            if "_fhir" in self.config["task"]:
                fhir_train_dataset = self.get_samples_dataset("train")
                fhir_val_dataset = self.get_samples_dataset("test")
            if "_documents" in self.config["task"]:
                doc_train_dataset, doc_val_dataset = self.get_document_dataset()

            train_dataset = self.unite_datasets(fhir_train_dataset, doc_train_dataset)
            val_dataset = self.unite_datasets(fhir_val_dataset, doc_val_dataset)

            assert (
                len(train_dataset) > 0
            ), "Something went wrong with the generation of the samples."
            logger.info(f"Total samples: {len(train_dataset)+len(val_dataset)}")

            train_dataset = self.tokenize(train_dataset)
            val_dataset = self.tokenize(val_dataset)
            logger.info("Starting pre-training...")

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
            eval_dataset=val_dataset if val_dataset else None,
            # compute_metrics=self.compute_metrics,
            callbacks=[TrainingLossLoggingCallback],
        )
        torch.cuda.empty_cache()
        trainer.train()
        trainer.save_model(self.model_best_path)


def main(config):
    pretrainer = Pretrainer(config)
    # Usage example
    pretrainer.pretrain(
        num_train_epochs=config["num_train_epochs"],
    )
