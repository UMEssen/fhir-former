import json
import logging
import random
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from fhirformer.fhir.util import OUTPUT_FORMAT, check_and_read
from fhirformer.data_preprocessing.util import (
    validate_resources,
    get_column_map_txt_resources,
    print_data_info,
    get_patient_ids_lists,
)
from fhirformer.data_preprocessing.data_store import DataStore

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
        self.config = config
        self.index = None
        self.patient_ids_lists = []
        self.patient_ids = []
        self.resources_for_task = None
        self.filtered_text_sampling_column_maps = None

    def set_up(self, store_list: List[DataStore]) -> None:
        assert (
            self.resources_for_task is not None
        ), "self.resources_for_task must be set by the inheriting class"

        # filter column_maps by resources_with_date_column
        validate_resources(self.resources_for_task, self.config)

        self.filtered_text_sampling_column_maps = get_column_map_txt_resources(
            self.config, self.resources_for_task
        )

        self.patient_ids_lists = get_patient_ids_lists(store_list)
        # get some info
        pats_int = sum([len(x) for x in self.patient_ids_lists])
        print_data_info(pats_int, store_list)

    def _process_split(
        self,
        idx_range: Tuple[int, int],
        df_resources: dict,
        pat_df: pd.DataFrame,
        resources_with_date_column: dict,
    ) -> "DataStore":
        start_idx, end_idx = idx_range
        fraction_patient_ids = self.patient_ids[start_idx:end_idx]
        fraction_pat_df = pat_df[pat_df["patient_id"].isin(fraction_patient_ids)]

        fraction_df_resources = {}
        for resource, df in df_resources.items():
            fraction_df_resources[resource] = df[
                df["patient_id"].isin(fraction_patient_ids)
            ]

        return DataStore(
            patient_df=fraction_pat_df,
            resources=fraction_df_resources,
            date_columns=resources_with_date_column,
        )

    def get_source_data(
        self,
        num_splits: int = 1,
    ) -> List[DataStore]:
        data_stores = []
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
        self.patient_ids = pat_df["patient_id"].to_list()

        date_columns_dict = {}
        for key, value in self.config["text_sampling_column_maps"].items():
            date_columns_dict[key] = value["main_col"]

        splits = (
            calculate_splits(len(self.patient_ids), num_splits)
            if num_splits > 1
            else [(0, len(self.patient_ids))]
        )

        logger.info("Running patient data split")
        for start_idx, end_idx in tqdm(splits, desc="Processing patient data splits"):
            fraction_patient_ids = self.patient_ids[start_idx:end_idx]
            fraction_pat_df = pat_df[pat_df["patient_id"].isin(fraction_patient_ids)]

            fraction_df_resources = {}
            for resource, df in df_resources.items():
                fraction_df_resources[resource] = df[
                    df["patient_id"].isin(fraction_patient_ids)
                ]

            data_stores.append(
                DataStore(
                    patient_df=fraction_pat_df,
                    resources=fraction_df_resources,
                    date_columns=date_columns_dict,
                )
            )
        return data_stores

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
        return ", ".join(tumors)

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

    def pat_history_to_string(self, pat_data: Dict) -> str:
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
                filtered_df.sort_values(by="date", inplace=True)

                filtered_df["resource"] = resource
                filtered_df["date"] = filtered_df["date"].dt.date
                filtered_dfs.append(filtered_df)

        if all(df.empty for df in filtered_dfs):
            return ""

        # if len(filtered_dfs) != len(
        #     self.filtered_text_sampling_column_maps
        # ):
        #     print(pat_data)
        #     return ""

        combined = pd.concat(filtered_dfs).reset_index(drop=True)
        combined.sort_values(by=["date", "resource"], inplace=True)
        combined.set_index("date", inplace=True)
        all_resources_string = self.df_to_string(combined)

        return all_resources_string.strip()

    @staticmethod
    def get_age_from_birth_date(birth_date: str) -> int:
        return (datetime.now() - pd.to_datetime(birth_date)).days // 365

    @staticmethod
    def df_to_string(com: pd.DataFrame):
        result_list = []
        current_date = None
        current_resource = None

        for row in com.itertuples():
            date = row.Index  # Access the 'date' field
            resource = row.resource  # Access the 'resource' field

            # Create a list of the non-null, non-"resource" values in the row
            content = [
                value
                for name, value in row._asdict().items()
                if not pd.isnull(value)
                and name != "Index"
                and name != "date"
                and name != "resource"
                and name != "patient_id"
            ]

            content_str = ", ".join(map(str, content))

            if date != current_date or resource != current_resource:
                if current_date is not None:
                    result_list.append("\n")
                result_list.append(f"{date} {resource}: {content_str}")
                current_date = date
                current_resource = resource
            else:
                result_list.append(f", {content_str}")

        result_str = "".join(result_list)
        return result_str

    @staticmethod
    def dict_to_string(d):
        return "\n".join(["\t".join([str(k), str(v)]) for k, v in d.items()])

    @staticmethod
    def process_patient(args):
        raise NotImplementedError("Please implement this for each specific task")

    @staticmethod
    def get_split_samples(sample_list: List[List[Dict]], split_ratio: float = 0.8):
        # Flatten the list
        flat_sample_list = [sample for sublist in sample_list for sample in sublist]

        # Remove empty dictionaries
        flat_sample_list = [x for x in flat_sample_list if len(x)]

        # Get unique patient IDs
        patient_ids = list(set([sample["patient_id"] for sample in flat_sample_list]))
        random.shuffle(patient_ids)

        # Split patient IDs into train and validation sets
        split_index = int(split_ratio * len(patient_ids))
        train_patients = patient_ids[:split_index]
        val_patients = patient_ids[split_index:]

        # Create train and validation samples
        train_samples = [
            sample
            for sample in flat_sample_list
            if sample["patient_id"] in train_patients
        ]
        val_samples = [
            sample
            for sample in flat_sample_list
            if sample["patient_id"] in val_patients
        ]

        return train_samples, val_samples

    def prepare(self, split_ratio: float) -> None:
        results = []
        for list_index, patient_ids in tqdm(
            enumerate(self.patient_ids_lists), desc="Overall progress of patient lists"
        ):
            if self.config["debug"]:
                numb_pats = round(len(patient_ids) * 0.001)
                patient_ids = patient_ids[:numb_pats]

            self.index = list_index

            # profiler = cProfile.Profile()
            # profiler.enable()

            with ProcessPoolExecutor(
                max_workers=30,
            ) as executor:
                results_iter = list(
                    tqdm(
                        executor.map(self.process_patient, patient_ids, chunksize=10),
                        total=len(patient_ids),
                        desc="Processing patients",
                    )
                )

            # profiler.disable()
            # stats = pstats.Stats(profiler).sort_stats("cumulative")
            # stats.sort_stats("cumulative").print_stats(
            #     300
            # )  # prints the top 10 cumulative time consuming functions calls
            # exit()

            # Remove empty lists
            results_iter = [x for x in results_iter if x]
            results.append(results_iter)

        # list of list to list
        results_list = [sample for sublist in results for sample in sublist]

        train_samples, val_samples = self.get_split_samples(results_list, split_ratio)
        all_samples_int = len(train_samples) + len(val_samples)

        if all_samples_int == 0:
            raise ValueError("No samples generated. Please check your data.")
        else:
            logger.info(f"Generated {all_samples_int} samples.")

        # Save the training samples
        with open(self.config["task_dir"] / "train.json", "w") as outfile:
            json.dump(train_samples, outfile, indent=4)

        # Save the validation samples
        with open(self.config["task_dir"] / "test.json", "w") as outfile:
            json.dump(val_samples, outfile, indent=4)
