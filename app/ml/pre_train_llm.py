import json
import logging
import os

import numpy as np
import torch
import wandb
from transformers import (
    TextDataset,
    DataCollatorForLanguageModeling,
    EvalPrediction,
    TrainerCallback,
    IntervalStrategy, EarlyStoppingCallback,
)
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForMaskedLM
from datasets import Dataset as HFDataset
from torch.nn import functional as F
from datasets import load_metric

os.environ["WANDB_LOG_MODEL"] = "end"


class TrainingLossLoggingCallback(TrainerCallback):
    def on_train_end(self, args, state, control, logs=None, **kwargs):
        if (
                state.global_step > 0
                and state.global_step % args.logging_steps == 0
                and "loss" in state.log_history[-1]
        ):
            # logs["train_loss"] = np.round(state.log_history[-1]["loss"], 4)
            wandb.log({logs["train_loss"]: np.round(state.log_history[-1]["loss"], 4)})

class PerplexityLoggingCallback(TrainerCallback):
    def __init__(self, compute_metrics_fn):
        super().__init__()
        self.compute_metrics_fn = compute_metrics_fn

    def on_evaluate(self, args, state, control, model, metrics, **kwargs):
        epoch = state.epoch
        if epoch > 0 and epoch % 5 == 0:
            perplexity = self.compute_metrics_fn(metrics)["eval_perplexity"]
            print(f"Perplexity at epoch {epoch}: {perplexity}")


class PerplexityEvaluationCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, subset_percentage=0.25):
        self.eval_dataset = eval_dataset.shuffle(seed=42)
        subset_len = int(subset_percentage * len(self.eval_dataset))
        self.eval_dataset = self.eval_dataset.select(range(subset_len))
        self.tokenizer = tokenizer
        self.perplexity_metric = load_metric("perplexity")

    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        # Perform evaluation on a subset of the validation dataset
        eval_dataloader = torch.utils.data.DataLoader(
            self.eval_dataset, batch_size=args.per_device_eval_batch_size
        )

        model.eval()
        all_logits = []
        all_labels = []
        for batch in eval_dataloader:
            with torch.no_grad():
                inputs = {k: v.to(args.device) for k, v in batch.items()}
                outputs = model(**inputs)
                logits = outputs.logits
                labels = inputs["labels"]
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        all_logits = np.concatenate(all_logits)
        all_labels = np.concatenate(all_labels)
        # Compute perplexity
        perplexity = self.perplexity_metric.compute(
            predictions=all_logits, references=all_labels
        )
        # Log perplexity
        print(f"Perplexity: {perplexity}")


class PretrainLongformer:
    def __init__(self, model_name, config, tokenizer=None):
        self.model_name = model_name
        self.config = config
        self.tokenizer = (
            tokenizer if tokenizer else AutoTokenizer.from_pretrained(model_name)
        )
        self.train_texts = self.get_text(is_train=True, num_samples=None)
        self.val_texts = self.get_text(is_train=False, num_samples=None)

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

    def pretrain(self, output_dir, num_train_epochs=2):
        logging.info("Starting pre-training...")
        wandb.init(
            tags=["baseline"],
            project=self.config["wandb_project_name"],
            name=f"{self.model_name.split('/')[-1]}_pretrain_{num_train_epochs}",
            mode="online",
            # mode="disabled",
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

        logging.info(f"total samples: {len(train_dataset)+len(val_dataset)}")

        # Define data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )

        # Define the training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=torch.cuda.device_count(),  #! per device
            per_device_eval_batch_size=1,  #! per device
            save_total_limit=2,
            load_best_model_at_end=True,
            eval_accumulation_steps=self.config["eval_accumulation_steps"],  # tune
            report_to="wandb",
            # report_to="none",
            evaluation_strategy="epoch",
            save_strategy=IntervalStrategy.EPOCH,
            fp16=True,
        )

        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=3,  # Number of steps with no improvement after which training will be stopped
            early_stopping_threshold=0.0,  # Minimum change in the monitored metric to be considered as an improvement
        )

        # Create the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=val_dataset if val_dataset else None,
            # compute_metrics=self.compute_metrics,
            callbacks=[TrainingLossLoggingCallback, early_stopping_callback],
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
        "/local/work/merengelke/ship_former/models",
        num_train_epochs=150,
    )


# Usage example
if __name__ == "__main__":
    main()
