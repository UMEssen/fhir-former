from pathlib import Path
from typing import Any, Generator, List
import datasets
import logging
from fhirformer.fhir.util import (
    OUTPUT_FORMAT,
    check_and_read,
    get_category_name,
    get_document_path,
)
from fhirformer.data_preprocessing.util import get_train_val_split

logger = logging.getLogger(__name__)


class DocumentConfig(datasets.BuilderConfig):
    def __init__(
        self,
        document_folder: Path,
        task_folder: Path,
        **kwargs: Any,
    ) -> None:
        super(DocumentConfig, self).__init__(**kwargs)
        self.document_folder = document_folder
        self.task_folder = task_folder


class DocumentDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    BUILDER_CONFIGS = [
        DocumentConfig(
            name="test",
            document_folder=Path.cwd(),
            task_folder=Path.cwd(),
        )
    ]
    RELEVANT_COLUMNS = [
        "diagnostic_report_id",
        "encounter_id",
        "category_display",
        "patient_id",
        "date",
    ]

    def _info(self) -> datasets.DatasetInfo:
        feature_dict = {
            "patient_id": datasets.Value("string"),
            "diagnostic_report_id": datasets.Value("string"),
            "encounter_id": datasets.Value("string"),
            "category": datasets.Value("string"),
            "category_display": datasets.Value("string"),
            "date": datasets.Value("string"),
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
        df = check_and_read(
            self.config.task_folder / f"diagnostic_report{OUTPUT_FORMAT}"
        )
        train_patients, val_patients = get_train_val_split(
            df["patient_id"].unique().tolist(),
            sample_by_letter=["0"],
            split_ratio=0.8,
        )
        train_df = df[df["patient_id"].isin(train_patients)]
        val_df = df[df["patient_id"].isin(val_patients)]

        train_df_dict = train_df.to_dict(orient="list")
        val_df_dict = val_df.to_dict(orient="list")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    relevant_column: train_df_dict[relevant_column]
                    for relevant_column in self.RELEVANT_COLUMNS
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    relevant_column: val_df_dict[relevant_column]
                    for relevant_column in self.RELEVANT_COLUMNS
                },
            ),
        ]

    def _generate_examples(
        self,
        diagnostic_report_id: List[str],
        encounter_id: List[str],
        category_display: List[List[str]],
        patient_id: List[str],
        date: List[str],
    ) -> Generator:
        logger.info(f"Generating splits for {len(diagnostic_report_id)} documents.")
        for i, document_name in enumerate(diagnostic_report_id):
            category_name = get_category_name(category_display[i])
            document_path = get_document_path(
                root_path=self.config.document_folder / category_name,
                filename=document_name + ".txt",
                folder_depth=64 // 8,
            )
            with document_path.open("r") as fp:
                text = fp.read()
            yield f"{document_name}", {
                "diagnostic_report_id": document_name,
                "encounter_id": encounter_id[i],
                "category_display": category_name,
                "patient_id": patient_id[i],
                "date": date[i],
                "text": text,
            }
