from typing import Dict, Union

import numpy as np
import torch
import wandb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from torch.nn import BCELoss
from transformers import Trainer, TrainingArguments

from fhirformer.helper.util import timed
from fhirformer.ml.callbacks import (
    BestScoreLoggingCallback,
    TrainingLossLoggingCallback,
)
from fhirformer.ml.downstream_task import DownstreamTask
from fhirformer.ml.patient_encounter_dataset import PatientEncounterDataset


class MainDiagnosisDataset(PatientEncounterDataset):
    def __init__(self, config, max_length=None, num_samples=None):
        super().__init__(config, max_length, num_samples)
        self.lb = LabelBinarizer()
        self.labels = self.lb.fit_transform([item["label"] for item in self.data])

        icds = [item["label"] for item in self.data]
        # Create a mapping of unique root ICD-10 codes to integers
        self.label_to_id = {}
        for icd in icds:
            if icd not in self.label_to_id:
                self.label_to_id[icd] = len(self.label_to_id)

    def __getitem__(self, idx) -> Dict[str, Union[int, str, torch.Tensor]]:
        return {
            **self.prepare_used_items(idx),
            "label_code": self.lb.classes_,
            "label_display": self.data[idx]["label"],
            "label": self.label_to_id[self.data[idx]["label"]],
            "decoded_labels": [
                self.lb.classes_[i]
                for i, label in enumerate(self.labels[idx])
                if label == 1
            ],
        }


class DS_ICD_Main(DownstreamTask):
    def __init__(
        self,
        config,
        model_checkpoint: str,
        batch_size: int = 2,
        epochs: int = 2,
        train_ratio: float = 0.8,
        prediction_cutoff: float = 0.5,
    ):
        super().__init__(
            config=config,
            dataset_class=MainDiagnosisDataset,
            dataset_args={"max_length": None, "num_samples": None},
            model_checkpoint=model_checkpoint,
            batch_size=batch_size,
            epochs=epochs,
            train_ratio=train_ratio,
            prediction_cutoff=prediction_cutoff,
        )
        label_freq = np.sum(
            self.dataset.labels, axis=0
        )  # sum over the column (each label)
        self.top10_classes = label_freq.argsort()[-10:][
            ::-1
        ]  # find top 10 most common classes

        self.model.config.problem_type = (
            "single_label_classification"  # specify problem type
        )

        self.model.loss = BCELoss()  # single class classification

        labels = [label for label in self.train_dataset[0]["label_codes"]]
        self.set_id_to_label(labels)

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        probabilities = self.softmax(logits)

        basic_metrics = self.metrics(
            predictions=predictions,
            labels=labels,
        )
        # Create a binary representation of the labels and probabilities
        unique_classes = np.unique(np.concatenate((labels, predictions)))
        lb = LabelBinarizer()
        lb.fit(unique_classes)
        binary_labels = lb.transform(labels)

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

        # Extract the top 10 labels and predictions
        mask = np.isin(predictions, self.top10_classes)
        labels_top10 = labels[mask]
        predictions_top10 = predictions[mask]

        metrics = {}
        for metric, value in basic_metrics.items():
            metrics["eval_" + metric] = value

        top_ten_metrics = self.metrics(
            predictions=predictions_top10, labels=labels_top10
        )
        for metric, value in top_ten_metrics.items():
            metrics["top10_" + metric] = value

        metrics["auc_roc"] = auc_roc
        metrics["macro_auc_pr"] = macro_auc_pr
        metrics["micro_auc_pr"] = micro_auc_pr

        for metric, value in metrics.items():
            metrics[metric] = round(value, 2) if value is not None else None

        # Log metrics to wandb
        wandb.log(metrics)

        return metrics

    def train(self) -> None:
        wandb.run.log_code(".")

        args = self.get_default_training_arguments()
        args["save_total_limit"] = 3
        training_args = TrainingArguments(**args)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[
                TrainingLossLoggingCallback,
                BestScoreLoggingCallback,
                self.early_stopping_callback(),
            ],
        )

        trainer.train()

        trainer.save_model(self.model_best_path)


@timed
def main(config):
    ds_icd_main = DS_ICD_Main(
        config,
        config["model_checkpoint"],
        epochs=config["train_epochs"],
    )
    ds_icd_main.train()
