import json
import logging
from pathlib import Path

import torch
import wandb
from datasets import load_dataset
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset

from fhirformer.helper.util import timed
from fhirformer.ml.util import get_evaluation_metrics

logger = logging.getLogger(__name__)


def convert_labels_to_ids(label, model_config):
    if isinstance(label, str):
        return model_config.label2id[label]
    elif isinstance(label, list):
        return [model_config.label2id[lab] for lab in label]
    else:
        return label


@timed
def main(config, num_wandb_samples: int = 10):
    split = (
        "test"
        if config["max_test_samples"] is None
        else f"test[:{config['max_test_samples']}]"
    )
    data = load_dataset(str(config["task_dir"] / "sampled"), split=split)

    model_path = Path(config["model_checkpoint"])
    if not (model_path / "tokenizer_config.json").exists():
        with (model_path / "config.json").open("rb") as f:
            model_name = json.load(f)["_name_or_path"]
        logger.info(f"Loading tokenizer from {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(str(model_path))

    pipe = pipeline(
        "text-classification",
        model=str(config["model_checkpoint"]),
        top_k=None,
        truncation=True,
        device=0 if torch.cuda.is_available() else -1,
    )
    # texts = [text for text in KeyDataset(data, "text")]
    labels = [label for label in KeyDataset(data, "labels")]
    single_label = not isinstance(labels[0], list)
    if single_label:
        binarizer = LabelBinarizer()
    else:
        binarizer = MultiLabelBinarizer(
            classes=[
                pipe.model.config.id2label[i]
                for i in range(len(pipe.model.config.label2id))
            ]
        )
    bin_labels = binarizer.fit_transform(labels)
    predictions = []
    probabilities = []
    for out in tqdm(pipe(KeyDataset(data, "text")), total=len(data), desc="Inference"):
        if single_label:
            predictions.append(out["label"])
            probabilities.append(out["score"])
        else:
            preds = []
            probas = []
            for res in out:
                if res["score"] > 0.5:
                    preds.append(res["label"])
                    probas.append(res["score"])
            predictions.append(preds)
            probabilities.append(probas)
    bin_predictions = binarizer.transform(predictions)
    basic_metrics = get_evaluation_metrics(
        labels=bin_labels, predictions=bin_predictions
    )

    if single_label:
        basic_metrics["auc_roc"] = roc_auc_score(bin_labels, probabilities)
        basic_metrics["auc_pr"] = average_precision_score(bin_labels, probabilities)

    metrics = {}
    for metric, value in basic_metrics.items():
        metrics["test_" + metric] = value

    logger.info(metrics)
    wandb.log(metrics)

    table = wandb.Table(
        # columns=["Predicted Label", "True Label", "Probabilities"],
        columns=["Text", "Predicted Label", "True Label", "Probabilities"],
    )

    for i in range(num_wandb_samples):
        table.add_data(
            # texts[i],
            str(predictions[i]),
            str(labels[i]),
            str(probabilities[i]),
        )

    wandb.log({f"inference_{num_wandb_samples}": table})
