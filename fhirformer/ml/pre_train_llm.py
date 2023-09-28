# import json
import json
import logging
import os

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

import wandb
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
        self.train_texts, self.val_texts = self.get_text(num_samples=None)
        self.model_best_path = config["model_dir"] / "best"
        self.model_best_path.mkdir(parents=True, exist_ok=True)
        self.wandb_project_name = "fhirformer_pretrain"

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

    def get_text(self, num_samples=None):
        with open(self.config["task_dir"] / "train.json", "r") as f:
            train_texts = json.load(f)[:num_samples] if num_samples else json.load(f)
        with open(self.config["task_dir"] / "test.json", "r") as f:
            val_texts = json.load(f)[:num_samples] if num_samples else json.load(f)
        return train_texts, val_texts

    def data_generator_train(self):
        for text in self.train_texts:
            yield {"text": text["text"]}

    def data_generator_val(self):
        for text in self.val_texts:
            yield {"text": text["text"]}

    def pretrain(self, num_train_epochs=2):
        logger.info("Starting pre-training...")
        wandb.init(
            tags=["baseline"],
            project=self.wandb_project_name,
            name=f"{self.config['model_name']}_{num_train_epochs}",
            mode="online",
            entity="ship-ai-autopilot",
        )
        wandb.run.log_code(".")

        # Create the model
        model = AutoModelForMaskedLM.from_pretrained(self.model_name)

        raw_dataset = HFDataset.from_generator(self.data_generator_train)
        train_dataset = raw_dataset.map(
            lambda examples: self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=None,  # , return_tensors="pt"
            )
        )

        raw_dataset = HFDataset.from_generator(self.data_generator_val)
        val_dataset = raw_dataset.map(
            lambda examples: self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=None,  # , return_tensors="pt"
            )
        )

        logger.info(f"Total samples: {len(train_dataset)+len(val_dataset)}")

        # Define data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )

        # Define the training arguments
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
            # report_to="none",
            evaluation_strategy="epoch",
            save_strategy=IntervalStrategy.EPOCH,
            fp16=False,
            metric_for_best_model="loss",
            greater_is_better=False,
        )

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

        # Train the model
        trainer.train()

        # Save the model
        trainer.save_model(self.model_best_path)


def main(config):
    pretrainer = Pretrainer(config)
    # Usage example
    pretrainer.pretrain(
        num_train_epochs=1000,
    )
