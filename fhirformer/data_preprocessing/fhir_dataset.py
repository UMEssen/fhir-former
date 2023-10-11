import json
import logging
from pathlib import Path
from typing import Any, Generator, List

import datasets

logger = logging.getLogger(__name__)


def load_and_transform(path: Path, max_samples: int = None):
    with open(path, "r") as f:
        samples = json.load(f)
    if max_samples:
        samples = samples[:max_samples]
        logger.info(f"Limiting samples to {max_samples}")
    patient_ids = [s["patient_id"] for s in samples]
    encounter_ids = [s["encounter_id"] for s in samples]
    texts = [s["text"] for s in samples]
    return {"patient_ids": patient_ids, "encounter_ids": encounter_ids, "texts": texts}


class FHIRConfig(datasets.BuilderConfig):
    def __init__(
        self,
        task_folder: Path,
        max_train_samples: int = None,
        max_test_samples: int = None,
        **kwargs: Any,
    ) -> None:
        super(FHIRConfig, self).__init__(**kwargs)
        self.task_folder = task_folder
        self.max_train_samples = max_train_samples
        self.max_test_samples = max_test_samples


class FHIRDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    BUILDER_CONFIGS = [
        FHIRConfig(
            # TODO: Would love this to be a wildcard but currently don't know how to do it
            name="pretrain_fhir",
            task_folder=Path.cwd(),
        )
    ]

    def _info(self) -> datasets.DatasetInfo:
        feature_dict = {
            "patient_id": datasets.Value("string"),
            "encounter_id": datasets.Value("string"),
            "text": datasets.Value("string"),
        }
        return datasets.DatasetInfo(
            description="TBD",
            features=datasets.Features(feature_dict),
            supervised_keys=None,
            homepage="TBD",
            citation="TBD",
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        train_samples = load_and_transform(
            self.config.task_folder / "train.json",
            max_samples=self.config.max_train_samples,
        )
        val_samples = load_and_transform(
            self.config.task_folder / "test.json",
            max_samples=self.config.max_test_samples,
        )
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs=train_samples,
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs=val_samples,
            ),
        ]

    def _generate_examples(
        self,
        encounter_ids: List[str],
        patient_ids: List[str],
        texts: List[str],
    ) -> Generator:
        for i in range(len(encounter_ids)):
            yield f"{patient_ids[i]}_{encounter_ids[i]}", {
                "patient_id": patient_ids[i],
                "encounter_id": encounter_ids[i],
                "text": texts[0],
            }
