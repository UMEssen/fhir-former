import json
import logging
import pickle
import random
from datetime import datetime
from functools import partial
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from fhirformer.data_preprocessing.data_store import DataStore
from fhirformer.data_preprocessing.util import (
    get_column_map_txt_resources,
    get_train_val_split,
    validate_resources,
)
from fhirformer.fhir.util import OUTPUT_FORMAT, check_and_read
from fhirformer.helper.util import get_nondependent_resources

logger = logging.getLogger(__name__)


def calculate_splits(total_patients, n):
    split_size = total_patients // n
    start_idx = 0
    end_idx = split_size
    splits = []
    for i in range(n):
        if i == n - 1:
            end_idx = total_patients
        splits.append((start_idx, end_idx))
        start_idx = end_idx
        end_idx += split_size
    return splits


class EncounterDatasetBuilder:
    def __init__(self, config):
        random.seed(42)
        self.num_processes = config["num_processes"]
        self.config = config
        self.sample_by_letter = None
        self.resources_for_task = get_nondependent_resources(self.config)
        validate_resources(self.resources_for_task, self.config)
        self.filtered_text_sampling_column_maps = get_column_map_txt_resources(
            self.config, self.resources_for_task
        )
        self.ds_folder = self.config["task_dir"] / "data_stores"
        self.set_up_data(num_splits=100)

    def set_up_data(
        self,
        num_splits: int = 1,
    ) -> None:
        num_stores = [
            1
            for i in range(1, num_splits + 1)
            if (self.ds_folder / f"datastore_{i}.pkl").exists()
        ]
        if sum(num_stores) == num_splits:
            logger.info(
                f"Skipping datastore creation, the splits are already stored in {self.ds_folder}"
            )
            return
        df_resources = {}
        # Read all resources first
        for resource, resource_dictionary in self.config[
            "text_sampling_column_maps"
        ].items():
            if self.config["task"] + "_bracket_cols" in resource_dictionary:
                columns_list = resource_dictionary[
                    self.config["task"] + "_bracket_cols"
                ]
            else:
                columns_list = resource_dictionary["bracket_cols"]
            columns_list.append("patient_id")
            columns_list.append(resource_dictionary["main_col"])
            logger.info(f"Reading and processing {resource}...")

            df = check_and_read(self.config["task_dir"] / f"{resource}{OUTPUT_FORMAT}")

            if self.config["task"] + "_drop_duplicates" in resource_dictionary:
                drop_cols = resource_dictionary[
                    self.config["task"] + "_drop_duplicates"
                ]
            elif "drop_duplicates" in resource_dictionary:
                drop_cols = resource_dictionary["drop_duplicates"]
            else:
                drop_cols = None

            if drop_cols:
                df.drop_duplicates(subset=drop_cols, inplace=True)

            # Filter columns
            df = df[columns_list]
            df_resources[resource] = df

        logger.info("Reading and processing patient...")
        pat_df = check_and_read(self.config["task_dir"] / f"patient{OUTPUT_FORMAT}")
        patient_ids = pat_df["patient_id"].unique().tolist()
        random.seed(42)
        random.shuffle(patient_ids)

        date_columns_dict = {}
        for key, value in self.config["text_sampling_column_maps"].items():
            date_columns_dict[key] = value["main_col"]

        splits = (
            calculate_splits(len(patient_ids), num_splits)
            if num_splits > 1
            else [(0, len(patient_ids))]
        )

        logger.info("Running patient data split")
        logger.info(
            f"Overall patients to process: "
            f"{len(patient_ids)} divided in {len(splits)}. "
            f"Split to patient ratio: {len(patient_ids) / len(splits)}"
        )
        self.ds_folder.mkdir(parents=True, exist_ok=True)

        for i, (start_idx, end_idx) in enumerate(
            tqdm(splits, desc="Processing patient data splits"),
            start=1,
        ):
            fraction_patient_ids = patient_ids[start_idx:end_idx]
            fraction_pat_df = pat_df[pat_df["patient_id"].isin(fraction_patient_ids)]

            fraction_df_resources = {}
            for resource, df in df_resources.items():
                fraction_df_resources[resource] = df[
                    df["patient_id"].isin(fraction_patient_ids)
                ]

            ds = DataStore(
                patient_df=fraction_pat_df,
                patient_list=fraction_patient_ids,
                resources=fraction_df_resources,
                date_columns=date_columns_dict,
            )
            with (self.ds_folder / f"datastore_{i}.pkl").open("wb") as of:
                pickle.dump(ds, of)

        logger.info(f"Finished dividing patient data into {len(splits)} splits.")

    def pat_df_to_string(self, patient_df: pd.DataFrame) -> str:
        # Patient corner
        pat_dict = {
            "age": self.get_age_from_birth_date(patient_df["birth_date"].values[0]),
            "gender": patient_df["sex"].values[0],
            "insurance_type": patient_df["insurance_type"].values[0],
        }
        return self.dict_to_string(pat_dict)

    @staticmethod
    def get_icd_version(condition: pd.DataFrame) -> str:
        if not condition["icd_version"].empty:
            # TODO: Why do we even put it if it's always the last one?
            return condition["icd_version"].iloc[-1]
        return "unknown"

    @staticmethod
    def get_tumors(episode_of_care: pd.DataFrame) -> str:
        tumors = []
        for row in episode_of_care.itertuples():
            tumor_str = None
            if row.first_diagnosis_date and row.treatment_program:
                tumor_str = f"{row.first_diagnosis_date} {row.treatment_program}"
            elif row.treatment_program:
                tumor_str = row.treatment_program
            if tumor_str:
                tumors.append(tumor_str)
        combined = ", ".join(tumors)
        if len(combined) > 0:
            return f"Tumor history: {combined}\n\n"
        else:
            return ""

    def enc_to_string(self, enc: Any) -> str:
        enc_dict = {}
        for col, name in [
            ("start", "Beginn"),
            ("type_display", "Kontakt Art"),
            ("v3_act_code", "Art"),
            ("fachabteilungsschluessel", "Fachabteilungsschlüssel"),
            ("aufnahmeanlass_display", "Aufnahmeanlass"),
        ]:
            if hasattr(enc, col):
                enc_dict[name] = getattr(enc, col)
        return self.dict_to_string(enc_dict)

    @staticmethod
    def filter_data(df: pd.DataFrame, columns_map: Dict[str, str]) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        if not all(col in df.columns for col in columns_map.keys()):
            raise ValueError(
                f"Columns {columns_map.keys()} not found in dataframe columns {df.columns}"
            )

        filtered_df = df[list(columns_map.keys())]
        return filtered_df.rename(columns=columns_map)

    def pat_history_to_string(
        self, pat_data: Dict, remove_duplicates: bool = True
    ) -> str:
        filtered_dfs = []

        for resource, config in self.filtered_text_sampling_column_maps.items():
            main_col = config.get("main_col", None)
            if self.config["task"] + "_bracket_cols" in config:
                bracket_cols = config[self.config["task"] + "_bracket_cols"]
            else:
                bracket_cols = config.get("bracket_cols", None)

            if resource == "condition" and "icd_version" in bracket_cols:
                bracket_cols.remove("icd_version")

            filtered_df = self.filter_data(
                pat_data.get(resource, pd.DataFrame()),
                {col: col for col in [main_col] + bracket_cols},
            )

            assert isinstance(
                filtered_df, pd.DataFrame
            ), "filtered_df must be a pd.DataFrame"

            if not filtered_df.empty:
                filtered_df[main_col] = pd.to_datetime(
                    filtered_df[main_col], format="%Y-%m-%d", utc=True, errors="coerce"
                )

                if (
                    not pd.api.types.is_datetime64_any_dtype(filtered_df[main_col])
                    or filtered_df[main_col].isna().any()
                ):
                    logger.error(
                        f"date column is not datetime type: {filtered_df[main_col]}"
                    )
                    filtered_df = filtered_df[~filtered_df[main_col].isna()]

                filtered_df.rename(columns={main_col: "date"}, inplace=True)
                filtered_df.sort_values(by="date", ascending=False, inplace=True)

                filtered_df["resource"] = resource
                filtered_df["date"] = filtered_df["date"].dt.date
                filtered_dfs.append(filtered_df)

        if all(df.empty for df in filtered_dfs):
            return ""

        combined = pd.concat(filtered_dfs).reset_index(drop=True)
        combined.sort_values(by=["date", "resource"], inplace=True)

        result_list = combined.groupby(["date", "resource"]).apply(
            partial(self.group_resources, remove_duplicates=remove_duplicates)
        )
        all_resources_string = "\n".join(result_list)

        return all_resources_string.strip()

    @staticmethod
    def get_age_from_birth_date(birth_date: str) -> int:
        return (datetime.now() - pd.to_datetime(birth_date)).days // 365

    @staticmethod
    def group_resources(df: pd.DataFrame, remove_duplicates: bool = True):
        relevant_info = []
        for row_dict in df.to_dict(orient="records"):
            relevant_info.append(
                ", ".join(
                    [
                        str(value)
                        for name, value in row_dict.items()
                        if not pd.isnull(value)
                        and name not in {"date", "resource", "patient_id"}
                    ]
                )
            )

        if remove_duplicates:
            relevant_counts = {x: relevant_info.count(x) for x in relevant_info}
            relevant_info = [
                relevant if count == 1 else f"{count}x {relevant}"
                for relevant, count in relevant_counts.items()
            ]

        date = df["date"].iloc[0]
        resource = df["resource"].iloc[0]
        return f"{date} {resource}: {'; '.join(relevant_info)}"

    @staticmethod
    def dict_to_string(d):
        return "\n".join(["\t".join([str(k), str(v)]) for k, v in d.items()])

    @staticmethod
    def process_patient(patient_id: str, datastore: DataStore) -> List[Dict]:
        raise NotImplementedError("Please implement this for each specific task")

    def get_split_samples(self, sample_list: List[Dict], split_ratio: float = 0.8):
        # Get unique patient IDs
        train_patients, val_patients = get_train_val_split(
            [sample["patient_id"] for sample in sample_list],
            sample_by_letter=self.sample_by_letter,
            split_ratio=split_ratio,
        )

        # Create train and validation samples
        train_samples = [
            sample for sample in sample_list if sample["patient_id"] in train_patients
        ]
        val_samples = [
            sample for sample in sample_list if sample["patient_id"] in val_patients
        ]

        return train_samples, val_samples

    def global_multiprocessing(self):
        """
        This is horrible, I know. This function will get repeated in all files, because of multiple reasons:
        1. There is no way to share a global variable between this class and each child class as far as I can see.
        Using global variables is already a bad solution, but I don't see any other way to do this.
        2. I have tried to share the DataStore using a NamespaceProxy, which is supposed to be the right way to do this,
        but it was blocking and just not working.
        I have tried something like this:
        https://stackoverflow.com/questions/72798554/how-to-use-multiprocessing-to-share-a-large-database-among-processes
        https://stackoverflow.com/questions/26499548/accessing-an-attribute-of-a-multiprocessing-proxy-of-a-class/68123850#68123850
        but it was still very very slow.
        """
        raise NotImplementedError("Please implement this for each specific task")

    def prepare(self, split_ratio: float) -> None:
        all_samples_file = self.config["task_dir"] / "all_samples.json"
        if not all_samples_file.exists():
            logger.info("Starting sample generation...")
            results = self.global_multiprocessing()
            # list of list to list
            sample_list = [sample for sublist in results for sample in sublist]
            # Flatten the list
            flat_sample_list = [sample for sublist in sample_list for sample in sublist]
            # Remove empty dictionaries
            flat_sample_list = [x for x in flat_sample_list if len(x)]

            if len(flat_sample_list) == 0:
                raise ValueError("No samples generated. Please check your data.")
            else:
                logger.info(f"Generated {len(flat_sample_list)} samples.")
            with open(all_samples_file, "w") as outfile:
                json.dump(flat_sample_list, outfile)
        else:
            logger.info("Loading samples from file...")
            with open(all_samples_file, "r") as infile:
                flat_sample_list = json.load(infile)

        train_samples, val_samples = self.get_split_samples(
            flat_sample_list, split_ratio
        )

        # Save the training samples
        with open(self.config["task_dir"] / "train.json", "w") as outfile:
            json.dump(train_samples, outfile)

        # Save the validation samples
        with open(self.config["task_dir"] / "test.json", "w") as outfile:
            json.dump(val_samples, outfile)
