import json
import logging
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from fhirformer.fhir.util import OUTPUT_FORMAT, check_and_read

logger = logging.getLogger(__name__)


def make_timezone_aware(value, default_timezone="UTC"):
    if isinstance(value, pd.Series):
        # Convert to datetime Series and ensure it's timezone-aware
        datetime_series = pd.to_datetime(value, utc=True).dt.tz_convert(
            default_timezone
        )
        return datetime_series
    elif isinstance(value, pd.Timestamp):
        # Check if the Timestamp is timezone-aware‚‚
        if not value.tz:
            return value.tz_localize(default_timezone)
        return value
    elif isinstance(value, str):
        timestamp = pd.to_datetime(value)
        if not timestamp.tz:
            return timestamp.tz_localize(default_timezone)
        return timestamp
    else:
        raise ValueError(
            f"Unsupported type {type(value)}. Expecting a pd.Series, pd.Timestamp, or str."
        )


def select_resources(
    resource_df: pd.DataFrame,
    column: str,
    patient_id: str,
    start_filter_date=None,
    end_filter_date=None,
):
    if len(resource_df) == 0:
        return resource_df

    condition = resource_df.patient_id == patient_id
    resource_tz_column = make_timezone_aware(resource_df[column])

    if start_filter_date:
        start_filter_date = make_timezone_aware(start_filter_date)
        condition &= resource_tz_column >= start_filter_date

    if end_filter_date:
        end_filter_date = make_timezone_aware(end_filter_date)
        condition &= resource_tz_column <= end_filter_date

    return resource_df.loc[condition]


@staticmethod
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


@dataclass
class DataStore:
    patient_df: pd.DataFrame
    resources: Dict[str, pd.DataFrame]
    date_columns: Dict[str, str]

    def filter_patient(
        self,
        patient_id: str,
        start_filter_date=None,
        end_filter_date=None,
        target_resource=None,
    ):
        filtered_patient = self.patient_df[self.patient_df.patient_id == patient_id]

        if target_resource:
            if target_resource not in self.resources:
                raise ValueError(
                    f"Resource {target_resource} not found in available resources."
                )

            filtered_resources = {
                target_resource: select_resources(
                    resource_df=self.resources[target_resource],
                    column=self.date_columns[target_resource],
                    patient_id=patient_id,
                    start_filter_date=start_filter_date,
                    end_filter_date=end_filter_date,
                )
            }
        else:
            filtered_resources = {
                resource_name: select_resources(
                    resource_df=resource_df,
                    column=self.date_columns[resource_name],
                    patient_id=patient_id,
                    start_filter_date=start_filter_date,
                    end_filter_date=end_filter_date,
                )
                for resource_name, resource_df in self.resources.items()
            }

        return DataStore(
            patient_df=filtered_patient,
            resources=filtered_resources,
            date_columns=self.date_columns,
        )


class EncounterDatasetBuilder:
    def __init__(self, config):
        random.seed(42)
        self.config = config

        # Somehow this is super low wtf
        # self.filtered_column_map_txt_resources = get_column_map_txt_resources(
        #     config, resources_for_task
        # )

        # # this has proven to be the fastest way to process patients so far
        # self.store_list_global = self.get_source_data(
        #     split_data=True,
        #     n=2,
        # )

        # # pass the fucking patient ids in a list each
        # self.patient_ids_lists = get_patient_ids_lists(self.store_list_global)

        # # get some info
        # pats_int = sum([len(x) for x in self.patient_ids_lists])
        # get_data_info(pats_int, self.store_list_global)
        # self.index = None

        # # filter column_maps by resources_with_date_column
        # validate_resources(resources_for_task, self.config)

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
        split_data: bool = False,
        n: int = 1,
    ) -> List["DataStore"]:
        data_stores = []
        df_resources = {}

        # Read all resources first
        for resource, value in self.config["text_sampling_column_maps"].items():
            columns_list = []
            for _, v2 in value.items():
                if isinstance(v2, list):
                    columns_list.extend(v2)
                else:
                    columns_list.append(v2)
            columns_list.append("patient_id")

            df = check_and_read(self.config["task_dir"] / f"{resource}{OUTPUT_FORMAT}")

            # Do the data post-processing here
            if self.config["task"] == "ds_image" or self.config["task"] == "pretrain":
                if resource == "condition":
                    df.drop_duplicates(
                        subset=["patient_id", "condition_id"], inplace=True
                    )
                elif resource == "imaging_study":
                    df.drop_duplicates(
                        subset=["patient_id", "imaging_study_id"], inplace=True
                    )
                    if self.config["task"] == "ds_image":
                        columns_list.extend(["procedure_code"])
                elif resource == "procedure":
                    df.drop_duplicates(
                        subset=["patient_id", "procedure_id"], inplace=True
                    )
                elif resource == "encounter":
                    df.drop_duplicates(subset=["patient_id", "id"], inplace=True)
                elif resource == "service_request":
                    cats_to_drop = [
                        x
                        for x in df.category_display.dropna().unique()
                        if "labor" in x.lower()
                        or "Imaging" in x
                        or "radio" in x.lower()
                        or "Röntgen" in x
                    ]
                    df = df[~df.category_display.isin(cats_to_drop)]

            if self.config["task"] == "ds_main_icd":
                logger.warning(
                    "You need to add the column that identifies the Main DRG here"
                )

            # Filter columns
            df = df[columns_list]
            df_resources[resource] = df

        pat_df = check_and_read(self.config["task_dir"] / f"patient{OUTPUT_FORMAT}")
        self.patient_ids = [pat for pat in pat_df["patient_id"]]

        date_columns_dict = {}
        for key, value in self.config["text_sampling_column_maps"].items():
            date_columns_dict[key] = value["main_col"]

        splits = (
            calculate_splits(len(self.patient_ids), n)
            if split_data
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

    def get_icd_version(self, condition: pd.DataFrame) -> str:
        if not condition["icd_version"].empty:
            return condition["icd_version"].iloc[-1]
        return "unknown"

    def enc_to_string(self, enc: Any) -> str:
        # assert isinstance(enc, pd.Series), f"enc must be a pd.Series {type(enc)}"

        columns = [
            "v3_act_code",
            "type_display",
            "fachabteilungsschluessel",
            "start",
            "aufnahmeanlass_display",
        ]

        # Check if all columns exist in the Pandas Series
        if all(col in enc for col in columns):
            enc_dict = {
                "Beginn": enc.start,
                "Kontarkt Art": enc.type_display,
                "Art": enc.v3_act_code,
                "Fachabteilungsschluessel": enc.fachabteilungsschluessel,
                "Aufnahmeanlass": enc.aufnahmeanlass_display,
            }
            return self.dict_to_string(enc_dict)
        else:
            # Handle the case where some columns are missing
            return "Some required columns are missing in the input Pandas Series."

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

        for resource, config in self.filtered_column_map_txt_resources.items():
            main_col = config.get("main_col", None)
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

        if self.config["debug"] and len(filtered_dfs) != len(
            self.filtered_column_map_txt_resources
        ):
            return ""

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
                if pd.notna(value)
                and name != "Index"
                and name != "date"
                and name != "resource"
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
            if self.config["debug"] or True:
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
