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
            self.filter_conditions()
        elif resource == "encounter":
            self.filter_encounter()
        elif resource == "biologically_derived_product":
            self.filter_bdp()
        elif resource == "patient":
            self.split_patients()
            self.filter_patient_info()
        elif resource == "procedure":
            self.filter_procedures()
        elif resource == "imaging_study":
            self.filter_imaging_studies()
        elif resource == "observation":
            self.filter_observation()
        elif resource == "episode_of_care":
            self.filter_episode_of_care()
        elif resource == "service_request":
            self.filter_service_request_pyrate()
        elif resource == "diagnostic_report":
            self.filter_diagnostic_report()
        elif resource == "medication":
            self.filter_medication()
        else:
            raise NotImplementedError(f"Resource {resource} not supported")

    # TODO: Datetime filtering for each metrics resource
    @staticmethod
    def filter_date(
        start: datetime, end: datetime, resource: pd.DataFrame, date_col: str
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

        with (self.config["data_dir"] / "patient_ids.pkl").open("wb") as of:
            pickle.dump(patients_ids, of)

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
            joined.drop(
                labels=["insurance_type", "birth_date", "sex", "deceased_date"],
                axis=1,
                inplace=True,
            )

        return joined

    def basic_filtering(
        self, name: str, output_name: str = None, save: bool = True, is_patient_df=False
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
        self.basic_filtering("encounter")

    def filter_medication(self):
        if (df := self.basic_filtering("medication", save=False)) is None:
            return
        store_df(df, self.config["task_dir"] / f"medication{OUTPUT_FORMAT}")

    def filter_diagnostic_report(self):
        if (df := self.basic_filtering("diagnostic_report", save=False)) is None:
            return
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

        store_df(df, self.config["task_dir"] / f"diagnostic_report{OUTPUT_FORMAT}")

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

    def filter_procedures(self) -> None:
        output_path = self.config["task_dir"] / f"procedure{OUTPUT_FORMAT}"
        if (pros := self.basic_filtering("procedure", save=False)) is None:
            return

        pros_filtered = pros.dropna(subset=["encounter_id"])

        pros_filtered = pros_filtered[
            (pros_filtered["status"] == "completed")
            | (pros_filtered["status"] == "in-progress")
            | (pros_filtered["status"] == "preparation")
        ]

        pros_filtered["start"] = pros_filtered[
            "effectivedatetimestart_v1"
        ].combine_first(pros_filtered["effectivedatetimestart_v2"])
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

        # Filtering by date of config
        pros_filtered["start"] = col_to_datetime(pros_filtered["start"])
        pros_filtered["end"] = col_to_datetime(pros_filtered["end"])
        store_df(pros_filtered, output_path)

    def filter_conditions(self) -> None:
        output_path = self.config["task_dir"] / f"condition{OUTPUT_FORMAT}"
        if (df_cond := self.basic_filtering("condition", save=False)) is None:
            return

        logging.info(f"Number of conditions before filtering: {len(df_cond)}")

        df_cond["code_diagnosis_type"] = reduce_cardinality(
            df_cond["code_diagnosis_type"], take_first=True
        )
        df_cond["code_diagnosis_display"] = reduce_cardinality(
            df_cond["code_diagnosis_display"], take_first=True
        )

        df_cond = df_cond.dropna(subset=["encounter_id", "patient_id", "condition_id"])

        logging.info(f"Number of conditions after filtering: {len(df_cond)}")
        store_df(df_cond, output_path)

    def filter_imaging_studies(self):
        output_path = self.config["task_dir"] / f"imaging_study{OUTPUT_FORMAT}"
        if (df_img := self.basic_filtering("imaging_study", save=False)) is None:
            return
        df_img.rename(columns={"id": "imaging_study_id"}, inplace=True)
        df_img.drop_duplicates(subset=["imaging_study_id"], inplace=True)
        store_df(df_img, output_path)

    def filter_observation(self):
        output_path = self.config["task_dir"] / f"observation{OUTPUT_FORMAT}"
        if (df_obs := self.basic_filtering("observation", save=False)) is None:
            return

        df_obs.dropna(subset=["value_quantity", "value_unit"], inplace=True)
        df_obs["code"] = reduce_cardinality(df_obs["code"], take_first=True)
        df_obs["display"] = reduce_cardinality(df_obs["display"], take_first=True)

        store_df(df_obs, output_path)

    def filter_episode_of_care(self):
        output_path = self.config["task_dir"] / f"episode_of_care{OUTPUT_FORMAT}"
        if (df_eoc := self.basic_filtering("episode_of_care", save=False)) is None:
            return
        df_eoc["treatment_program"] = df_eoc["treatment_program"].map(TUMOR_TYPE_MAP)
        store_df(df_eoc, output_path)

    def filter_service_request_pyrate(self):
        output_path = self.config["task_dir"] / f"service_request{OUTPUT_FORMAT}"
        if (df := self.basic_filtering("service_request", save=False)) is None:
            return

        df = df[df["status"].isin(["active", "completed", "draft", "unknown"])]
        df.dropna(subset=["patient_id"], inplace=True)

        df["category_code"] = reduce_cardinality(df["category_code"], take_first=True)
        df["category_display"] = reduce_cardinality(
            df["category_display"], take_first=True
        )

        store_df(df, output_path)
