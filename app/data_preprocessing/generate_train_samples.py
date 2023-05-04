import json
import multiprocessing as mp
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any
import pandas as pd
from pathlib import Path
import pickle
from concurrent.futures import ProcessPoolExecutor

from numpy import dtype
from tqdm import tqdm


@dataclass
class DataStore:
    pro: Optional[pd.DataFrame]
    enc: pd.DataFrame
    con: Optional[pd.DataFrame]
    pat: pd.DataFrame

    def select_resources(
        self, resource_df: pd.DataFrame, column: str, patient_id: str, filter_date
    ):
        if len(resource_df) == 0:
            return resource_df

        return resource_df.loc[
            (resource_df.patient_id == patient_id)
            & (pd.to_datetime(resource_df[column]) >= filter_date)
        ]

    def filter_patient(self, patient_id: str, filter_date=None):
        if filter_date:
            date = pd.to_datetime(filter_date)

            return DataStore(
                self.select_resources(
                    resource_df=self.pro,
                    column="procedure_start",
                    patient_id=patient_id,
                    filter_date=date,
                ),
                self.select_resources(
                    resource_df=self.enc,
                    column="encounter_start",
                    patient_id=patient_id,
                    filter_date=date,
                ),
                self.select_resources(
                    resource_df=self.con,
                    column="condition_date",
                    patient_id=patient_id,
                    filter_date=date,
                ),
                self.pat[self.pat.patient_id == patient_id],
            )

        return DataStore(
            self.pro[self.pro.patient_id == patient_id],
            self.enc[self.enc.patient_id == patient_id].sort_values(
                ["start"], inplace=False
            ),
            self.con[self.con.patient_id == patient_id],
            self.pat[self.pat.patient_id == patient_id],
        )


def make_store_global(store: Any) -> None:
    global store_global
    store_global = store


class EncounterTokenBuilder:
    def __init__(self, config):
        self.config = config
        self.store = self.get_source_data()

    def get_source_data(self) -> DataStore:
        # Procedure
        pro = pd.read_feather(Path(self.config["procedure_path_filtered"]))
        # Encounter
        enc = pd.read_feather(Path(self.config["encounter_path_filtered"]))
        # Conditions
        con = pd.read_feather(Path(self.config["condition_path_filtered"]))
        # Patients
        pats = pd.read_feather(Path(self.config["patient_path_filtered"]))
        return DataStore(pro=pro, enc=enc, con=con, pat=pats)

    @staticmethod
    def get_age_from_birth_date(birth_date: str) -> datetime:
        return (datetime.now() - pd.to_datetime(birth_date)).days // 365

    @staticmethod
    def df_to_string(df, main_col, bracket_cols):
        s = ""
        for _, row in df.iterrows():
            main_value = row[main_col].values[0]
            bracket_values = ", ".join([str(row[col]) for col in bracket_cols])
            s += f"{main_value} ({bracket_values})\n"
        return s.strip()

    @staticmethod
    def dict_to_string(d):
        return "\n".join(["\t".join([str(k), str(v)]) for k, v in d.items()])

    @staticmethod
    def process_patient(args):
        (pat,) = args
        store = store_global
        pat_data = DataStore.filter_patient(store, patient_id=pat)
        sample_list = []
        # pat_data = self.store.filter_patient(patient_id=pat)
        if len(pat_data.enc) == 0 or len(pat_data.con) == 0:
            return []
        for _, enc in pat_data.enc.iterrows():
            pro = pat_data.pro[
                pat_data.pro.encounter_id == enc.encounter_id
            ].reset_index(drop=True)
            con = pat_data.con[
                pat_data.con.encounter_id == enc.encounter_id
            ].reset_index(drop=True)
            if len(con) == 0:
                return []

            con["condition_date"] = [
                x.date() for x in pd.to_datetime(con["condition_date"])
            ]

            con_label = con.loc[con["code_med_hauptdiagnose"]].reset_index(drop=True)
            con_train = con.loc[~con["code_med_hauptdiagnose"]].reset_index(drop=True)

            # todo challenge on ship-prod
            if len(con_label) != 1:
                return []

            # Calculate duration in days and append to 'enc_clean'
            duration = (enc["end"] - enc["start"]).days
            enc = pd.concat(
                [enc, pd.Series({"duratioin hospital": duration})], axis=0
            )

            enc_clean = enc[
                ~enc.index.isin(
                    [
                        "encounter_id",
                        "type",
                        "status",
                        "patient_id",
                        "practicioner_id",
                        "end",
                        "start",
                    ]
                )
            ]

            enc_dict = enc_clean.to_dict()
            pat_dict = {
                "age": EncounterTokenBuilder.get_age_from_birth_date(
                    pat_data.pat["birthDate"].values[0]
                ),
                "gender": pat_data.pat["gender"].values[0],
                "insurance_type": pat_data.pat["insurance_type"].values[0],
            }

            version = (
                con_label["icd_version"].values[0] if len(con_label) else "unknown"
            )

            pro["procedure_start"] = [
                x.date() for x in pd.to_datetime(pro["procedure_start"])
            ]

            assert len(con_label) == 1, "Only one hauptdiagnose per encounter allowed"

            version_str = EncounterTokenBuilder.dict_to_string({"version": version})
            pat_str = EncounterTokenBuilder.dict_to_string(pat_dict)
            pro_str = EncounterTokenBuilder.df_to_string(
                pro, main_col=["code"], bracket_cols=["code_display", "procedure_start"]
            )
            con_str = EncounterTokenBuilder.df_to_string(
                con_train,
                main_col=["icd_code"],
                bracket_cols=["icd_display", "condition_date"],
            )
            enc_str = EncounterTokenBuilder.dict_to_string(enc_dict)

            text = f"Patient_Metadata:\n{pat_str}\n\nProcedures:\n{pro_str}\n\nConditions (version {version_str}):\n{con_str}\n\nEncounter:\n{enc_str}"

            sample_list.append(
                {
                    "patient_id": str(pat),
                    "encounter_id": str(enc["encounter_id"]),
                    "label": con_label["icd_root_code"].values[0],
                    "text": text,
                }
            )
            return sample_list

    def build_encounter_token(self) -> None:
        print("starting pool")

        args = [(pat,) for pat in self.store.pat.patient_id[:5000]]

        with ProcessPoolExecutor(
            max_workers=40, initializer=make_store_global, initargs=(self.store,)
        ) as executor:
            results_iter = list(
                tqdm(
                    executor.map(
                        EncounterTokenBuilder.process_patient,
                        args,
                    ),
                    total=len(args),
                    desc="Processing patients",
                )
            )

        results = list(results_iter)
        sample_list = [sample for sublist in results for sample in sublist]

        # print(
        #     f"Number of features: {len(enc_dict.keys()) + len(pat_dict.keys()) + len(pro.columns) + len(con_train.columns)}"
        # )

        # Writing the list of dicts to a file
        with open(self.config["sample_path"], "w") as outfile:
            json.dump(sample_list, outfile, indent=4)


def main(config) -> None:
    if not os.path.exists(config["sample_path"]) or config["reload_cache"] or True:
        enc_token_builder = EncounterTokenBuilder(config)
        enc_token_builder.build_encounter_token()


if __name__ == "__main__":
    main()
