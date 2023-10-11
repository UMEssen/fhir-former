from typing import Dict, Union

import numpy as np
import torch
import wandb
from scipy.special import expit
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn import BCEWithLogitsLoss

from fhirformer.helper.util import timed
from fhirformer.ml.downstream_task import DownstreamTask
from fhirformer.ml.patient_encounter_dataset import PatientEncounterDataset


class MultiLabelDataset(PatientEncounterDataset):
    def __init__(self, config, max_length=None, num_samples=None):
        super().__init__(config, max_length, num_samples)

        possible_labels = [item["labels"] for item in self.data]
        # Create a mapping of unique root ICD-10 codes to integers
        self.config = config
        self.mlb = MultiLabelBinarizer()
        self.labels = self.mlb.fit_transform(possible_labels)
        self.num_classes = len(self.mlb.classes_)

    def __getitem__(self, idx) -> Dict[str, Union[int, str, torch.Tensor]]:
        return {
            **self.prepare_used_items(idx),
            "label_codes": self.mlb.classes_,
            "labels": torch.Tensor(self.labels[idx]),
            "decoded_labels": [
                self.mlb.classes_[i]
                for i, label in enumerate(self.labels[idx])
                if label == 1
            ],
        }


class MultiLabelTrainer(DownstreamTask):
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
            dataset_class=MultiLabelDataset,
            dataset_args={
                "config": config,
                "max_length": None,
                "num_samples": self.config["max_train_samples"],
            },
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
            "multi_label_classification"  # specify problem type
        )
        self.model.loss = BCEWithLogitsLoss()  # specify loss function for multi-label

        labels = [label for label in self.train_dataset[0]["label_codes"]]
        self.set_id_to_label(labels)

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        probabilities = expit(logits)  # sigmoid function
        predictions = (probabilities > self.prediction_cutoff).astype(
            int
        )  # threshold at 0.5logits, labels = eval_pred

        # Compute metrics for top 10 classes
        labels_top10 = labels[:, self.top10_classes]
        predictions_top10 = predictions[:, self.top10_classes]

        basic_metrics = self.metrics(predictions=predictions, labels=labels)
        metrics = {}
        for metric, value in basic_metrics.items():
            metrics["eval_" + metric] = value

        top_ten_metrics = self.metrics(
            predictions=predictions_top10, labels=labels_top10
        )
        for metric, value in top_ten_metrics.items():
            metrics["top10_" + metric] = value

        for metric, value in metrics.items():
            metrics[metric] = round(value, 2)

        # Log metrics to wandb
        wandb.log(metrics)

        return metrics


@timed
def main(config):
    ds_multi = MultiLabelTrainer(
        config, config["model_checkpoint"], epochs=config["num_train_epochs"]
    )
    ds_multi.train()
