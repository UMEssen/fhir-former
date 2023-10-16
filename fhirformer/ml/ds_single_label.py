from typing import Dict, Union

import numpy as np
import torch
import wandb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from torch.nn import BCELoss

from fhirformer.helper.util import timed
from fhirformer.ml.downstream_task import DownstreamTask
from fhirformer.ml.patient_encounter_dataset import PatientEncounterDataset


class SingleLabelDataset(PatientEncounterDataset):
    def __init__(self, config, max_length=None, num_samples=None):
        super().__init__(config, max_length, num_samples)
        self.lb = LabelBinarizer()
        self.labels = self.lb.fit_transform([item["labels"] for item in self.data])
        self.num_classes = 2
        self.problem_type = "single_label_classification"

        # Create a mapping of unique root ICD-10 codes to integers
        self.label_to_id = {
            icd: idx
            for idx, icd in enumerate(set(item["labels"] for item in self.data))
        }

    def __getitem__(self, idx) -> Dict[str, Union[int, str, torch.Tensor]]:
        return {
            **self.prepare_used_items(idx),
            "label_codes": self.lb.classes_,
            "label_display": self.data[idx]["labels"],
            "labels": self.label_to_id[self.data[idx]["labels"]],
        }


class SingleLabelTrainer(DownstreamTask):
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
            dataset_class=SingleLabelDataset,
            dataset_args={"config": config, "max_length": None, "num_samples": None},
            model_checkpoint=model_checkpoint,
            batch_size=batch_size,
            epochs=epochs,
            train_ratio=train_ratio,
            prediction_cutoff=prediction_cutoff,
        )
        self.model.loss = BCELoss()  # single class classification

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        probabilities = self.softmax(logits)
        basic_metrics = self.metrics(
            predictions=predictions, labels=labels, single_label=True
        )
        # Safely compute AUC-ROC and AUC-PR
        auc_roc, auc_pr = 0, 0
        try:
            auc_roc = roc_auc_score(labels, probabilities)
            auc_pr = average_precision_score(labels, probabilities)
        except ValueError:
            pass

        metrics = {}
        for metric, value in basic_metrics.items():
            metrics["eval_" + metric] = value

        metrics.update({"auc_roc": round(auc_roc, 2), "auc_pr": round(auc_pr, 2)})

        # Log metrics to wandb
        wandb.log(metrics)

        return metrics


@timed
def main(config):
    single_label = SingleLabelTrainer(
        config,
        config["model_checkpoint"],
        epochs=config["num_train_epochs"],
    )
    single_label.train()
