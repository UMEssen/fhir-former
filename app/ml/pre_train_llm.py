import json
import logging

import numpy as np
import torch
import wandb
from transformers import (
    TextDataset,
    DataCollatorForLanguageModeling,
    EvalPrediction,
    TrainerCallback,
    IntervalStrategy,
)
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import Dataset as HFDataset
from torch.nn import functional as F


def compute_metrics(eval_pred: EvalPrediction, batch_size=256):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_batches = int(np.ceil(logits.shape[0] / batch_size))
    perplexity = 0.0

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        # Convert numpy array to tensor and move to the correct device
        logits_batch = torch.from_numpy(logits[start_idx:end_idx]).to(device)
        labels_batch = torch.from_numpy(labels[start_idx:end_idx]).to(device)

        # Now calculate the loss
        masked_lm_loss = (
            F.cross_entropy(
                logits_batch.view(-1, logits_batch.size(-1)), labels_batch.view(-1)
            )
            .detach()
            .cpu()
            .numpy()
        )

        perplexity += np.exp(masked_lm_loss)

    perplexity /= num_batches
    return {"perplexity": perplexity}


class TrainingLossLoggingCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        # torch.cuda.empty_cache()
        last_loss_log = [x for x in state.log_history if "loss" in x]
        if last_loss_log:
            last_loss_log = last_loss_log[-1]

            if state.global_step > 0 and state.global_step % args.logging_steps == 0:
                if logs is None:
                    logs = {}

                logs["train_loss"] = np.round(last_loss_log["loss"], 4)


class PretrainLongformer:
    def __init__(self, model_name, config, tokenizer=None, block_size=128):
        self.model_name = model_name
        self.config = config
        self.block_size = block_size
        self.tokenizer = (
            tokenizer if tokenizer else AutoTokenizer.from_pretrained(model_name)
        )
        self.train_texts = self.get_text(is_train=True, num_samples=None)
        self.val_texts = self.get_text(is_train=False, num_samples=None)

    def get_text(self, is_train=True, num_samples=None):
        if is_train:
            with open(self.config["train_sample_path"], "r") as f:
                texts = json.load(f)[:num_samples] if num_samples else json.load(f)
        else:
            with open(self.config["val_sample_path"], "r") as f:
                texts = json.load(f)[:num_samples] if num_samples else json.load(f)
        return texts

    def data_generator_train(self):
        for text in self.train_texts:
            yield {"text": text["text"]}

    def data_generator_val(self):
        for text in self.val_texts:
            yield {"text": text["text"]}

    def prepare_dataset_legacy_hug(self):
        logging.info("Preparing dataset...")
        return TextDataset(
            tokenizer=self.tokenizer,
            file_path=self.config["train_sample_path"],
            block_size=self.block_size,
        )

    def pretrain(self, output_dir, num_train_epochs=2):
        logging.info("Starting pre-training...")
        wandb.init(
            tags=["baseline"],
            project="ship_llm",
            name=f"{self.model_name.split('/')[-1]}_pretrain_{num_train_epochs}",
            mode="online",
            # mode= "disabled",
        )
        wandb.run.log_code(".")

        # Create the model
        model = AutoModelForMaskedLM.from_pretrained(self.model_name)

        raw_dataset = HFDataset.from_generator(self.data_generator_train)
        train_dataset = raw_dataset.map(
            lambda examples: self.tokenizer(
                examples["text"], truncation=True, padding="max_length", max_length=None
            )
        )

        raw_dataset = HFDataset.from_generator(self.data_generator_val)
        val_dataset = raw_dataset.map(
            lambda examples: self.tokenizer(
                examples["text"], truncation=True, padding="max_length", max_length=None
            )
        )

        # Define data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )

        # Define the training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            save_total_limit=2,
            load_best_model_at_end=True,
            report_to="wandb",
            eval_accumulation_steps=50,
            # report_to="none",
            evaluation_strategy="epoch",
            save_strategy=IntervalStrategy.EPOCH,
            fp16=True,
        )

        # Create the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset if val_dataset else None,
            compute_metrics=compute_metrics,
            callbacks=[TrainingLossLoggingCallback],
        )

        # Train the model
        trainer.train()

        # Save the model
        trainer.save_model(output_dir)


def main(config):
    model_checkpoint = "bert-base-uncased"

    pretrainer = PretrainLongformer(model_checkpoint, config)
    # Usage example
    pretrainer.pretrain(
        "/local/work/merengelke/icd_pred/results/pretrained_models/bert",
        num_train_epochs=60,
    )


# Usage example
if __name__ == "__main__":
    main()
