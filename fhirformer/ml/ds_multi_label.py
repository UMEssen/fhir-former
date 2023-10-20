import logging

import numpy as np
import wandb
from datasets import interleave_datasets
from scipy.special import expit
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn import BCEWithLogitsLoss

from fhirformer.helper.util import timed
from fhirformer.ml.downstream_task import DownstreamTask
from fhirformer.ml.util import get_evaluation_metrics

logger = logging.getLogger(__name__)


class MultiLabelTrainer(DownstreamTask):
    def __init__(
        self,
        config,
        prediction_cutoff: float = 0.5,
    ):
        self.lb = MultiLabelBinarizer()
        self.top10_classes = None
        super().__init__(
            config=config,
            problem_type="multi_label_classification",
            prediction_cutoff=prediction_cutoff,
        )
        if self.config["task"] == "ds_image":
            # Balance dataset by checking how many labels there are
            self.make_train_dataset_balanced()

        # Set up the model parameters
        self.set_up_models(num_labels=len(self.lb.classes_))
        self.model.loss = BCEWithLogitsLoss()  # specify loss function for multi-label
        self.set_id_to_label(self.lb.classes_)
        self.tokenize_datasets()

    def set_up_dataset_labels(self):
        # This function gets called within init of the parent class
        labels = self.lb.fit_transform(self.dataset["labels"])
        label_freq = np.sum(labels, axis=0)  # sum over the column (each label)
        # find top 10 most common classes
        self.top10_classes = label_freq.argsort()[-10:][::-1]
        # Transform the labels to one-hot encoding
        self.dataset = self.dataset.rename_column("labels", "decoded_labels")
        self.dataset = self.dataset.map(
            lambda x: {
                "labels": self.lb.transform([x["decoded_labels"]])[0].astype(np.float32)
            },
            desc="Transforming labels to one-hot encoding",
        )

    def count_labels(self):
        frequencies = np.sum(self.train_dataset["labels"], axis=1)
        zero_count = np.sum(frequencies == 0)
        more_count = np.sum(frequencies > 0)
        return more_count, zero_count

    def make_train_dataset_balanced(self):
        more_count, zero_count = self.count_labels()
        logger.info(
            f"Balancing the dataset which has {zero_count} with 0 labels "
            f"and {more_count} more than 0 labels."
        )
        negatives = self.train_dataset.filter(lambda x: sum(x["labels"]) == 0)
        positives = self.train_dataset.filter(lambda x: sum(x["labels"]) > 0)
        self.train_dataset = interleave_datasets(
            [positives, negatives], probabilities=[0.7, 0.3]
        )
        more_count, zero_count = self.count_labels()
        logger.info(
            f"The dataset was balanced with {zero_count} with 0 labels "
            f"and {more_count} with more than zero labels."
        )

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        probabilities = expit(logits)  # sigmoid function
        # threshold at 0.5
        predictions = (probabilities > self.prediction_cutoff).astype(int)

        # Compute metrics for top 10 classes
        labels_top10 = labels[:, self.top10_classes]
        predictions_top10 = predictions[:, self.top10_classes]

        basic_metrics = get_evaluation_metrics(predictions=predictions, labels=labels)
        metrics = {}
        for metric, value in basic_metrics.items():
            metrics["eval_" + metric] = value

        top_ten_metrics = get_evaluation_metrics(
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
    ds_multi = MultiLabelTrainer(config=config)
    ds_multi.train()
