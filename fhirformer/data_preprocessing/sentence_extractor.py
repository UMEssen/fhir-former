import json
import logging
import math
import multiprocessing
import random
import re

from tqdm import tqdm

from fhirformer.data_preprocessing.constants import SAMPLE_BY_LETTER
from fhirformer.data_preprocessing.deduplication import run_deduplication
from fhirformer.data_preprocessing.util import get_train_val_split
from fhirformer.fhir.util import (
    FOLDER_DEPTH,
    OUTPUT_FORMAT,
    check_and_read,
    get_category_name,
    get_document_path,
    store_df,
)

logger = logging.getLogger(__name__)


class SentenceExtractor:
    def __init__(
        self,
        config,
    ):
        self.config = config
        self.sentence_folder = self.config["task_dir"] / "sentences"
        self.sentence_folder.mkdir(exist_ok=True, parents=True)

    def extract(self):
        logger.info("Extracting the sentences...")
        document_splits_train = (
            self.config["task_dir"] / f"documents_train{OUTPUT_FORMAT}"
        )
        document_splits_val = self.config["task_dir"] / f"documents_val{OUTPUT_FORMAT}"
        if document_splits_train.exists():
            logger.info("Loading documents splits from disk.")
            train_df = check_and_read(document_splits_train)
            val_df = check_and_read(document_splits_val)
        else:
            logger.info("Creating documents splits...")
            df = check_and_read(
                self.config["task_dir"] / f"diagnostic_report{OUTPUT_FORMAT}"
            )
            train_patients, val_patients = get_train_val_split(
                df["patient_id"].unique().tolist(),
                sample_by_letter=SAMPLE_BY_LETTER,
                split_ratio=0.8,
            )
            train_df = df[df["patient_id"].isin(train_patients)]
            val_df = df[df["patient_id"].isin(val_patients)]
            store_df(train_df, document_splits_train)
            store_df(val_df, document_splits_val)

        train_df_dict = train_df.to_dict(orient="records")
        val_df_dict = val_df.to_dict(orient="records")

        random.seed(42)
        for phase, data, max_samples in [
            ("train", train_df_dict, self.config["max_train_samples"]),
            ("val", val_df_dict, self.config["max_test_samples"]),
        ]:
            with multiprocessing.Pool(processes=self.config["num_processes"]) as pool:
                logger.info(f"Generating samples for {len(data)} documents.")
                if max_samples is not None:
                    selected_data = [
                        data[i]
                        for i in random.sample(
                            range(0, len(train_df_dict)), max_samples
                        )
                    ]
                else:
                    selected_data = data
                with (self.sentence_folder / f"{phase}.jsonl").open("w") as of:
                    for res in tqdm(
                        pool.imap_unordered(self.split_documents, selected_data),
                        total=len(data),
                        desc=f"Splitting documents for {phase}",
                    ):
                        for r in res:
                            json.dump(r, of)
                            of.write("\n")

    def split_documents(self, inp, chunk_size=50, overlap_words=10):
        diagnostic_report_id = inp["diagnostic_report_id"]
        encounter_id = inp["encounter_id"]
        original_category_display = inp["original_category_display"]
        patient_id = inp["patient_id"]
        # date = inp["date"]

        document_name = diagnostic_report_id
        category_name = get_category_name(original_category_display)
        document_path = get_document_path(
            root_path=self.config["data_dir"] / "documents" / category_name,
            filename=document_name + ".txt",
            folder_depth=FOLDER_DEPTH,
        )
        if not document_path.exists():
            return []
        with document_path.open("r") as fp:
            text = fp.read()
        base_dict = {
            "diagnostic_report_id": document_name,
            "encounter_id": encounter_id,
            "category_display": category_name,
            "patient_id": patient_id,
            # "date": date,
        }
        records = []
        words = [(match.start(), match.end()) for match in re.finditer(r"\s+", text)]
        ranges = [
            (
                max((i * chunk_size) - overlap_words, 0),
                min(
                    ((i + 1) * chunk_size) + overlap_words,
                    len(words) - 1,
                ),
            )
            for i in range(0, math.ceil(len(words) / chunk_size))
        ]

        for init_range, end_range in ranges:
            init_index = words[init_range][0]
            end_index = words[end_range][1]
            new_dict = base_dict.copy()
            new_dict["text"] = text[init_index:end_index].strip()
            new_dict["start_index"] = init_index
            new_dict["end_index"] = end_index
            records.append(new_dict)

        return records

    def deduplicate(self):
        logger.info("Starting deduplication...")
        (self.config["task_dir"] / "sentences_deduplicated").mkdir(
            parents=True, exist_ok=True
        )
        run_deduplication(
            input_dir=self.config["task_dir"] / "sentences",
            column="text",
            cache_dir=self.config["task_dir"] / "dedup_cache",
            ngram_size=13,
            num_perm=128,
            threshold=0.85,
            min_ngram_size=3,
            output_file=self.config["task_dir"]
            / "sentences_deduplicated"
            / "sentences_train_deduplicated.jsonl",
            split="train",
        )


def main(config):
    sentence_extractor = SentenceExtractor(config)
    if (
        not (sentence_extractor.sentence_folder / "train.jsonl").exists()
        or not (sentence_extractor.sentence_folder / "val.jsonl").exists()
    ):
        sentence_extractor.extract()
    sentence_extractor.deduplicate()
