import logging

import evaluate
import numpy as np
import wandb
from datasets import interleave_datasets
from sklearn.preprocessing import LabelBinarizer
from torch.nn import BCELoss

from fhirformer.helper.util import timed
from fhirformer.ml.downstream_task import DownstreamTask

logger = logging.getLogger(__name__)


class SingleLabelTrainer(DownstreamTask):
    def __init__(
        self,
        config,
        prediction_cutoff: float = 0.5,
    ):
        self.lb = LabelBinarizer()
        super().__init__(
            config=config,
            problem_type="single_label_classification",
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
        metric_accuracy = evaluate.load("accuracy")
        metric_precision = evaluate.load("precision")
        metric_recall = evaluate.load("recall")
        metric_f1 = evaluate.load("f1")
        metric_auc_roc = evaluate.load("roc_auc")

        results = {}
        results.update(
            metric_accuracy.compute(predictions=predictions, references=labels)
        )
        results.update(
            metric_precision.compute(
                predictions=predictions, references=labels, average="macro"
            )
        )
        results.update(
            metric_recall.compute(
                predictions=predictions, references=labels, average="macro"
            )
        )
        results.update(
            metric_f1.compute(
                predictions=predictions, references=labels, average="macro"
            )
        )

        results.update(
            metric_auc_roc.compute(prediction_scores=predictions, references=labels)
        )

        for key, value in results.items():
            results[key] = round(value, 4)

        # Log metrics to wandb
        wandb.log(results)

        return results


@timed
def main(config):
    single_label = SingleLabelTrainer(config)
    single_label.train()
