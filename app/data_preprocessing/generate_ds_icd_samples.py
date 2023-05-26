import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any
import pandas as pd
from pathlib import Path
import pickle
from concurrent.futures import ProcessPoolExecutor
from app.data_preprocessing.generate_pre_train_samples import EncounterTokenBuilder as ETB

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
        """
        Process a single patient
        Restrictions:
            - Patient must have at least one encounter
            - Patient must have at least one condition
            - Encounter must have at least one condition
            - Duration of encounter must be at least 3 days
            - One "hauptdiagnose" must be present
            - When sampling and all conditions and procedures happened before the sampling date -> skip
            - label may not start with "GB"
        Sampling:
            - Sampling starts one day after encounter start
            - Step size is one day

        """

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

            # Meta Constants
            # Patient corner
            pat_dict = {
                "age": EncounterTokenBuilder.get_age_from_birth_date(
                    pat_data.pat["birthDate"].values[0]
                ),
                "gender": pat_data.pat["gender"].values[0],
                "insurance_type": pat_data.pat["insurance_type"].values[0],
            }

            # Encounter corner
            # Calculate duration in days and append to 'enc_clean'
            duration = (enc["end"] - enc["start"]).days
            if duration <= 2:
                return []

            enc = pd.concat([enc, pd.Series({"duration hospital": duration})], axis=0)

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
            enc_str = EncounterTokenBuilder.dict_to_string(enc_dict)
            pat_str = EncounterTokenBuilder.dict_to_string(pat_dict)

            # Condition corner
            con["condition_date"] = [
                x.date() for x in pd.to_datetime(con["condition_date"])
            ]
            con_label = con.loc[con["code_med_hauptdiagnose"]].reset_index(drop=True)
            # con_train = con.loc[~con["code_med_hauptdiagnose"]].reset_index(drop=True)

            # todo challenge on ship-prod
            if len(con_label) != 1:
                return []
            assert len(con_label) == 1, "Only one hauptdiagnose per encounter allowed"

            version = (
                con_label["icd_version"].values[0] if len(con_label) else "unknown"
            )

            # Procedure corner
            pro["procedure_start"] = [
                x.date() for x in pd.to_datetime(pro["procedure_start"])
            ]
            version_str = EncounterTokenBuilder.dict_to_string(
                {"ICD-Version:": version}
            )

            # Generate multiple samples from each encounter
            for date in pd.date_range(
                enc.start + pd.Timedelta(days=1), enc.end - pd.Timedelta(days=1)
            ):
                date = date.date()
                date_filtered_con = con[con.condition_date <= date]
                date_filtered_pro = pro[pro.procedure_start <= date]

                label_con = con[con.condition_date > date]
                # Skip if no new conditions or procedures -> TODO extend once you have new data sources
                if date > pd.to_datetime(
                    date_filtered_con["condition_date"].max()
                ) or date > pd.to_datetime(date_filtered_pro["procedure_start"].max()):
                    continue
                labels = list(
                    set(
                        [
                            label
                            for label in label_con.icd_code.values.tolist()
                            if not label.startswith("GB")
                            and not label.startswith("GROUPBOX")
                            and not label.startswith("PATSTURZ")
                        ]
                    )
                )
                # Skip if no labels
                if not labels:
                    continue

                date_filtered_pro = date_filtered_pro.rename(
                    columns={"code_display": "description", "procedure_start": "date"}
                )
                date_filtered_con = date_filtered_con.rename(
                    columns={
                        "icd_code": "code",
                        "icd_display": "description",
                        "condition_date": "date",
                    }
                )

                combined = pd.concat(
                    [
                        date_filtered_con[["code", "description", "date"]],
                        date_filtered_pro[["code", "description", "date"]],
                    ]
                ).reset_index(drop=True)
                combined.sort_values(by="date", inplace=True)

                combined_str = EncounterTokenBuilder.df_to_string(
                    combined, main_col=["code"], bracket_cols=["description", "date"]
                )

                sample_date = EncounterTokenBuilder.dict_to_string(
                    {"sample_date": date}
                )

                if len(combined_str) == 0:
                    continue

                text = f"Patient_Metadata:\n{pat_str}\n\nEncounter:\n{enc_str}\n\n{version_str}\n{sample_date}\n\nProcedures and Condition:\n{combined_str}"

                unique_labels = list(set([x.split(".")[0] for x in labels]))
                if len(unique_labels) == 0:
                    logging.error("No labels")
                sample_list.append(
                    {
                        "patient_id": str(pat),
                        "encounter_id": str(enc["encounter_id"]),
                        "text": text,
                        "label": list(set([x.split(".")[0] for x in labels])),
                    }
                )


            # todo run this if more resources are added to make sure the labels are valid
            # samples = pd.DataFrame(sample_list)
            # duplicate_patients = samples[
            #     samples.duplicated(subset=["patient_id"], keep=False)
            # ]
            # if len(duplicate_patients) > 0:
            #     for pat in duplicate_patients["patient_id"].drop_duplicates():
            #         # print(pat)
            #         x = samples[samples["patient_id"] == pat]["label"].values
            #         for x_ in x:
            #             for x__ in x_:
            #                 if len(x__) > 4:
            #                     print(x)
            return sample_list

    def build_encounter_token(self) -> None:
        print("starting pool")

        args = [(pat,) for pat in self.store.pat.patient_id[:]]

        with ProcessPoolExecutor(
            max_workers=30, initializer=make_store_global, initargs=(self.store,)
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
        patient_ids = list(set([sample['patient_id'] for sample in sample_list]))

        patient_ids_validation = pd.read_pickle(self.config["pre_uq_pats_for_ds_task"])
        patient_ids_train = [x for x in patient_ids if x not in patient_ids_validation]

        print(f"Ratio of patients in train set: {len(patient_ids_train)/len(patient_ids)}")
        if not 0.75 < len(patient_ids_train)/len(patient_ids) < 0.85:
            raise ValueError("Ratio of patients in train/val set is not between 0.75 and 0.85")
            # if it is less you can't really do anything
            # if it is more you can add more patients to the train set

        train_samples, test_samples = ETB.get_split_samples(sample_list)

        # Save the training samples
        with open(self.config["ds_icd_train_sample_path"], "w") as outfile:
            json.dump(train_samples, outfile, indent=4)

        # Note: Test samples are not used in downstream tasks and never touched
        with open(self.config["ds_icd_test_sample_path"], "w") as outfile:
            json.dump(test_samples, outfile, indent=4)


def main(config) -> None:
    if (
        not os.path.exists(config["ds_icd_train_sample_path"])
        or config["reload_cache"]
    ):
        enc_token_builder = EncounterTokenBuilder(config)
        enc_token_builder.build_encounter_token()


if __name__ == "__main__":
    main()
