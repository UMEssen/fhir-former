import logging

import numpy as np
import wandb
from datasets import interleave_datasets
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from torch.nn import BCELoss

from fhirformer.helper.util import timed
from fhirformer.ml.downstream_task import DownstreamTask
from fhirformer.ml.util import get_evaluation_metrics

logger = logging.getLogger(__name__)


class SingleLabelTrainer(DownstreamTask):
    def __init__(
        self,
        config,
        train_ratio: float = 0.8,
        prediction_cutoff: float = 0.5,
    ):
        self.lb = LabelBinarizer()
        super().__init__(
            config=config,
            problem_type="single_label_classification",
            train_ratio=train_ratio,
            prediction_cutoff=prediction_cutoff,
        )
        # Balance dataset by checking how many labels there are
        self.make_train_dataset_balanced()

        # Set up the model parameters
        self.set_up_models(num_labels=2)
        self.model.loss = BCELoss()  # single class classification
        self.tokenize_datasets()

    def set_up_dataset_labels(self):
        # This function gets called within init of the parent class
        _ = self.lb.fit_transform(self.dataset["labels"])
        mapping_dict = {value: i for i, value in enumerate(self.lb.classes_)}
        # Transform the labels to one-hot encoding
        self.dataset = self.dataset.rename_column("labels", "decoded_labels")
        self.dataset = self.dataset.map(
            lambda x: {"labels": mapping_dict[x["decoded_labels"]]},
            desc="Transforming labels to one-hot encoding",
        )

    def count_labels(self):
        pos_count = sum(self.train_dataset["labels"])
        neg_count = len(self.train_dataset["labels"]) - pos_count
        return pos_count, neg_count

    def make_train_dataset_balanced(self):
        pos_count, neg_count = self.count_labels()
        logger.info(
            f"Balancing the dataset which has {pos_count} positives and {neg_count} negatives."
        )
        positives = self.train_dataset.filter(lambda x: x["labels"] == 1)
        negatives = self.train_dataset.filter(lambda x: x["labels"] == 0)
        self.train_dataset = interleave_datasets([positives, negatives], seed=42)
        pos_count, neg_count = self.count_labels()
        logger.info(
            f"The dataset was balanced with "
            f"{pos_count} positives and {neg_count} negatives."
        )

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        probabilities = self.softmax(logits)
        predictions = np.argmax(probabilities, axis=-1)
        basic_metrics = get_evaluation_metrics(
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
    single_label = SingleLabelTrainer(config)
    single_label.train()
