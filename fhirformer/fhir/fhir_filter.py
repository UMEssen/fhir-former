import datetime
import logging
import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from fhirformer.data_preprocessing.constants import TUMOR_TYPE_MAP
from fhirformer.fhir.util import (
    OUTPUT_FORMAT,
    check_and_read,
    col_to_datetime,
    handle_empty_df,
    reduce_cardinality,
    store_df,
)

logger = logging.getLogger(__name__)

# Class FHIRExtractor


class FHIRFilter:
    def __init__(self, config):
        self.config = config

    def filter(self, resource: str):
        resource = resource.lower()
        if resource == "condition":
            self.filter_conditions_wrapper()
        elif resource == "encounter":
            self.filter_encounter()
        elif resource == "biologically_derived_product":
            self.filter_bdp()
        elif resource == "patient":
            self.split_patients()
            self.filter_patient_info()
        elif resource == "procedure":
            self.filter_procedures_wrapper()
        elif resource == "imaging_study":
            self.filter_imaging_studies_wrapper()
        elif resource == "observation":
            self.filter_observation_wrapper()
        elif resource == "episode_of_care":
            self.filter_episode_of_care_wrapper()
        elif resource == "service_request":
            self.filter_service_request_pyrate_wrapper()
        elif resource == "diagnostic_report":
            self.filter_diagnostic_report_wrapper()
        elif resource == "medication":
            self.filter_medication()
        else:
            raise NotImplementedError(f"Resource {resource} not supported")

    @staticmethod
    def filter_date(
        start: datetime.datetime,
        end: datetime.datetime,
        resource: pd.DataFrame,
        date_col: str,
    ) -> pd.DataFrame:
        df = resource[
            ((start <= resource[date_col]) & (resource[date_col] <= end))
        ].sort_values([date_col])

        return df

    def skip_filter(self, path: Path):
        if self.config["rerun_cache"] or not path.exists():
            return False
        else:
            return path.exists()

    def split_patients(self):
        input_path = self.config["data_dir"] / f"patient{OUTPUT_FORMAT}"
        output_path = self.config["data_dir"] / f"pretrain_patient_ids{OUTPUT_FORMAT}"
        if self.skip_filter(output_path):
            return
        pat_df = check_and_read(input_path)

        # only include patients that do have an encounter
        path = self.config["data_dir"] / f"encounter{OUTPUT_FORMAT}"
        encs = check_and_read(path)
        pat_df = pat_df[pat_df["patient_id"].isin(encs["patient_id"])]

        patients_ids = pat_df["linked_patient_id"].unique().tolist()

        logger.info(
            f"Number of patients before splitting into pretrain and downstream: {len(patients_ids)}"
        )

        # Save all patient IDs
        with (self.config["data_dir"] / "patient_ids.pkl").open("wb") as of:
            pickle.dump(patients_ids, of)

        # Check if pre-train workflow is enabled
        use_pretrain = self.config.get("use_pretrain_workflow", False)

        if use_pretrain:
            # Split patients into pretrain and downstream sets
            pretrain_pats = patients_ids[: len(patients_ids) // 2]
            downstream_pats = patients_ids[len(patients_ids) // 2 :]

            logger.info(
                f"Pre-train workflow enabled. There are {len(pretrain_pats)} "
                f"({len(pretrain_pats) / len(patients_ids):.2f}) patients for pretraining "
                f"and {len(downstream_pats)} ({len(downstream_pats) / len(patients_ids):.2f}) "
                f"patients for downstream tasks."
            )
        else:
            # Use all patients for downstream tasks when pre-train workflow is disabled
            pretrain_pats = pd.DataFrame()  # Empty DataFrame for pretrain
            downstream_pats = patients_ids  # All patients for downstream

            logger.info(
                f"Pre-train workflow disabled. Using all {len(downstream_pats)} patients "
                f"for downstream tasks."
            )

        # Save the pretrain and downstream patient IDs
        with (self.config["data_dir"] / "pretrain_patient_ids.pkl").open("wb") as of:
            pickle.dump(pretrain_pats, of)

        with (self.config["data_dir"] / "downstream_patient_ids.pkl").open("wb") as of:
            pickle.dump(downstream_pats, of)

    def filter_bdp(self):
        output_path = (
            self.config["task_dir"] / f"biologically_derived_product{OUTPUT_FORMAT}"
        )
        if (
            df := self.basic_filtering("biologically_derived_product", save=False)
        ) is None:
            return

        df.drop_duplicates(subset=["bdp_id", "service_request_id"], inplace=True)
        df = df[df["ausgabe_type"] == "AUSGABE"]
        df["ausgabe_datetime"] = pd.to_datetime(df["ausgabe_datetime"])
        store_df(df, output_path)

    @staticmethod
    def to_datetime(df, col_format):
        for k, v in tqdm(col_format.items(), desc="Converting DateTime"):
            df[k] = pd.to_datetime(df[k], format=v, utc=True, errors="coerce")
        return df

    @handle_empty_df(required_columns=["patient_id"])
    def filter_by_meta_patients(
        self, df: pd.DataFrame, is_patient_df: bool = False
    ) -> pd.DataFrame:
        # filtering out patients that are not in the patient table
        pats = check_and_read(self.config["data_dir"] / "patient_ids.pkl")
        meta_pats = check_and_read(self.config["data_dir"] / f"patient{OUTPUT_FORMAT}")

        filtered_meta = meta_pats[meta_pats["linked_patient_id"].isin(pats)]

        joined = pd.merge(
            left=df,
            right=filtered_meta,
            on="patient_id",
            how="inner",
            suffixes=("", "_meta"),
        )

        joined = joined[joined.columns.drop(list(joined.filter(regex="_meta")))]

        joined.dropna(subset=["patient_id"], inplace=True)
        joined.rename(
            columns={
                "linked_patient_id": "patient_id",
                "patient_id": "original_patient_id",
            },
            inplace=True,
        )

        if not is_patient_df:
            # Only drop columns if they exist
            cols_to_drop = ["insurance_type", "birth_date", "sex", "deceased_date"]
            existing_cols = [col for col in cols_to_drop if col in joined.columns]
            if existing_cols:
                joined.drop(labels=existing_cols, axis=1, inplace=True)

        return joined

    def basic_filtering(
        self,
        name: str,
        output_name: Optional[str] = None,
        save: bool = True,
        is_patient_df=False,
    ) -> Optional[pd.DataFrame]:
        output_name = name if output_name is None else output_name
        output_path = self.config["task_dir"] / f"{output_name}{OUTPUT_FORMAT}"
        if self.skip_filter(output_path):
            return None
        df = check_and_read(self.config["data_dir"] / f"{output_name}{OUTPUT_FORMAT}")
        df = self.filter_by_meta_patients(df, is_patient_df=is_patient_df)
        if save:
            store_df(df, output_path)
        else:
            return df

    def filter_encounter(self):
        # most filtering is already done in build_encounter
        if (df := self.basic_filtering("encounter", save=False)) is None:
            return

        # Convert dates to datetime with UTC timezone
        df["start"] = col_to_datetime(df["start"])
        df["end"] = col_to_datetime(df["end"])

        if self.config["live_inference"]:
            df.sort_values(by="start", inplace=True)
            df = df[df["start"] <= datetime.datetime.now(datetime.timezone.utc)]
            df = df[df["end"] >= datetime.datetime.now(datetime.timezone.utc)]
            df = df.groupby("patient_id").last().reset_index()

        store_df(df, self.config["task_dir"] / f"encounter{OUTPUT_FORMAT}")

    def filter_medication(self):
        if (df := self.basic_filtering("medication", save=False)) is None:
            return
        store_df(df, self.config["task_dir"] / f"medication{OUTPUT_FORMAT}")

    @handle_empty_df()
    def filter_diagnostic_report(self, df: pd.DataFrame) -> pd.DataFrame:
        df["date"] = df.apply(
            lambda x: (
                x["issued"]
                if pd.isnull(x["effective_datetime"])
                else x["effective_datetime"]
            ),
            axis=1,
        )

        # if title is a list take only the first
        df["title"] = df["title"].apply(lambda x: x[0] if isinstance(x, list) else x)
        df["original_category_display"] = df["category_display"]

        df["category"] = reduce_cardinality(df["category"], take_first=True)
        df["content_type"] = reduce_cardinality(df["content_type"], take_first=True)
        return df

    def filter_patient_info(self):
        output_path = self.config["task_dir"] / f"patient{OUTPUT_FORMAT}"
        if (
            df := self.basic_filtering("patient", save=False, is_patient_df=True)
        ) is None:
            return

        df.drop_duplicates(subset=["patient_id"], inplace=True)

        df["insurance_type"] = (
            df["insurance_type"]
            .map({"PKV": "privat", "GKV": "gesetzlich"})
            .fillna("unbekannt")
        )
        df.drop(columns=["original_patient_id"], inplace=True)

        store_df(df, output_path)

    @handle_empty_df(required_columns=["encounter_id"])
    def filter_procedures(self, pros: pd.DataFrame) -> pd.DataFrame:
        pros_filtered = pros.dropna(subset=["encounter_id"])
        pros_filtered = pros_filtered[
            (pros_filtered["status"] == "completed")
            | (pros_filtered["status"] == "in-progress")
            | (pros_filtered["status"] == "preparation")
        ]

        pros_filtered["start"] = pros_filtered[
            "effectivedatetimestart_v1"
        ].combine_first(pros_filtered["effectivedatetimeend_v2"])
        pros_filtered["end"] = pros_filtered["effectivedatetimeend_v1"].combine_first(
            pros_filtered["effectivedatetimeend_v2"]
        )

        pros_filtered = pros_filtered[
            [
                "procedure_id",
                "patient_id",
                "encounter_id",
                "code",
                "status",
                "version",
                "display",
                "start",
                "end",
                "location_id",
            ]
        ]

        # Convert dates to datetime with UTC timezone
        pros_filtered["start"] = col_to_datetime(pros_filtered["start"])
        pros_filtered["end"] = col_to_datetime(pros_filtered["end"])

        return pros_filtered

    @handle_empty_df(required_columns=["encounter_id", "patient_id", "condition_id"])
    def filter_conditions(self, df_cond: pd.DataFrame) -> pd.DataFrame:
        logging.info(f"Number of conditions before filtering: {len(df_cond)}")

        df_cond["code_diagnosis_type"] = reduce_cardinality(
            df_cond["code_diagnosis_type"], take_first=True
        )
        df_cond["code_diagnosis_display"] = reduce_cardinality(
            df_cond["code_diagnosis_display"], take_first=True
        )

        df_cond = df_cond.dropna(subset=["encounter_id", "patient_id", "condition_id"])

        logging.info(f"Number of conditions after filtering: {len(df_cond)}")
        return df_cond

    @handle_empty_df()
    def filter_imaging_studies(self, df_img: pd.DataFrame) -> pd.DataFrame:
        df_img.rename(columns={"id": "imaging_study_id"}, inplace=True)
        df_img.drop_duplicates(subset=["imaging_study_id"], inplace=True)
        return df_img

    @handle_empty_df(required_columns=["value_quantity", "value_unit"])
    def filter_observation(self, df_obs: pd.DataFrame) -> pd.DataFrame:
        df_obs.dropna(subset=["value_quantity", "value_unit"], inplace=True)
        df_obs["code"] = reduce_cardinality(df_obs["code"], take_first=True)
        df_obs["display"] = reduce_cardinality(df_obs["display"], take_first=True)
        return df_obs

    @handle_empty_df()
    def filter_episode_of_care(self, df_eoc: pd.DataFrame) -> pd.DataFrame:
        if "treatment_program" in df_eoc.columns:
            df_eoc["treatment_program"] = df_eoc["treatment_program"].map(
                TUMOR_TYPE_MAP
            )
        return df_eoc

    @handle_empty_df(required_columns=["patient_id", "status"])
    def filter_service_request(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["status"].isin(["active", "completed", "draft", "unknown"])]
        df.dropna(subset=["patient_id"], inplace=True)
        df["category_code"] = reduce_cardinality(df["category_code"], take_first=True)
        df["category_display"] = reduce_cardinality(
            df["category_display"], take_first=True
        )
        return df

    def filter_procedures_wrapper(self) -> None:
        output_path = self.config["task_dir"] / f"procedure{OUTPUT_FORMAT}"
        if (pros := self.basic_filtering("procedure", save=False)) is None:
            return
        result = self.filter_procedures(pros)
        store_df(result, output_path)

    def filter_conditions_wrapper(self) -> None:
        output_path = self.config["task_dir"] / f"condition{OUTPUT_FORMAT}"
        if (df_cond := self.basic_filtering("condition", save=False)) is None:
            return
        result = self.filter_conditions(df_cond)
        store_df(result, output_path)

    def filter_imaging_studies_wrapper(self):
        output_path = self.config["task_dir"] / f"imaging_study{OUTPUT_FORMAT}"
        if (df_img := self.basic_filtering("imaging_study", save=False)) is None:
            return
        result = self.filter_imaging_studies(df_img)
        store_df(result, output_path)

    def filter_observation_wrapper(self):
        output_path = self.config["task_dir"] / f"observation{OUTPUT_FORMAT}"
        if (df_obs := self.basic_filtering("observation", save=False)) is None:
            return
        result = self.filter_observation(df_obs)
        store_df(result, output_path)

    def filter_episode_of_care_wrapper(self):
        output_path = self.config["task_dir"] / f"episode_of_care{OUTPUT_FORMAT}"
        if (df_eoc := self.basic_filtering("episode_of_care", save=False)) is None:
            return
        result = self.filter_episode_of_care(df_eoc)
        store_df(result, output_path)

    def filter_service_request_pyrate_wrapper(self):
        output_path = self.config["task_dir"] / f"service_request{OUTPUT_FORMAT}"
        if (df := self.basic_filtering("service_request", save=False)) is None:
            return
        result = self.filter_service_request(df)
        store_df(result, output_path)

    def filter_diagnostic_report_wrapper(self):
        output_path = self.config["task_dir"] / f"diagnostic_report{OUTPUT_FORMAT}"
        if (df := self.basic_filtering("diagnostic_report", save=False)) is None:
            return
        result = self.filter_diagnostic_report(df)
        store_df(result, output_path)
