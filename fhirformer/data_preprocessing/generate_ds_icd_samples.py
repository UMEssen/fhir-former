import logging

import pandas as pd

from fhirformer.data_preprocessing.encounter_dataset_builder import (
    EncounterDatasetBuilder,
)

from fhirformer.data_preprocessing.util import skip_build, load_datastore
from tqdm import tqdm
from multiprocessing import Pool

logger = logging.getLogger()


class ICD10DatasetBuilder(EncounterDatasetBuilder):
    def __init__(self, config):
        super().__init__(config)
        # Here you can simply decide which resources you want to use
        self.resources_for_task = [
            "procedure",
            "condition",
            "imaging_study",
            "biologically_derived_product",
            "observation",
            "service_request",
            "medication",
            "episode_of_care",
            "diagnostic_report",
        ]
        self.set_up()

    def global_multiprocessing(self):
        results = []
        for datastore_path in tqdm(
            sorted(self.ds_folder.glob("datastore*.pkl")),
            desc="Overall progress of datastores",
        ):
            global datastore
            datastore = load_datastore(datastore_path)
            with Pool(
                processes=10,
            ) as executor:
                results_iter = list(
                    tqdm(
                        executor.imap_unordered(
                            self.process_patient,
                            datastore.patient_list,
                            chunksize=10,
                        ),
                        total=len(datastore.patient_list),
                        desc="Processing patients",
                    )
                )

            # Remove empty lists
            results_iter = [x for x in results_iter if x]
            results.append(results_iter)
            if self.config["debug"]:
                break
        return results

    def process_patient(self, patient_id):
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
        pat_data = datastore.filter_patient(patient_id=patient_id)

        sample_list = []
        if len(pat_data.enc) == 0 or len(pat_data.con) == 0:
            return []

        patient_metadata_str = (
            f"Patient metadata:\n{self.pat_df_to_string(pat_data.patient_df)}\n\n"
        )

        for _, enc in pat_data.enc.iterrows():
            resources_during_enc = pat_data.filter_patient(
                patient_id=patient_id,
                start_filter_date=enc["start"],
                end_filter_date=enc["end"],
            )
            if len(resources_during_enc["imaging_study"]) == 0:
                continue

            duration = (enc["end"] - enc["start"]).days
            if duration <= 2:
                continue

            # Generate multiple samples from each encounter
            for date in pd.date_range(
                enc.start + pd.Timedelta(days=1), enc.end - pd.Timedelta(days=1)
            ):
                resources_during_sample = pat_data.filter_patient(
                    patient_id=patient_id,
                    start_filter_date=enc.start,
                    end_filter_date=date,
                )

                if len(resources_during_sample["condition"]) == 0:
                    continue

                con_labels_df = pat_data.filter_patient(
                    patient_id=patient_id,
                    start_filter_date=date,
                    end_filter_date=enc.end,
                    target_resource="condition",
                )

                label_con = con_labels_df["icd_code"].tolist()

                # todo filter out conditions that start with "GB", "GROUPBOX", "PATSTURZ" before
                # todo make sure that there we have new conditions compared to the last sample

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

                pat_hist = self.pat_history_to_string(resources_during_enc)

                if not pat_hist:
                    continue

                text = (
                    f"{patient_metadata_str}"
                    f"Encounter:\n{self.enc_to_string(enc)}\n\n"
                    f"Duration: {duration}\n\n"
                    f"ICD Version: {self.get_icd_version(resources_during_enc['condition'])}\n\n"
                    f"Patient journey:\n{pat_hist}"
                )

                unique_labels = list(set([x.split(".")[0] for x in labels]))
                if len(unique_labels) == 0:
                    logger.error("No labels")
                sample_list.append(
                    {
                        "patient_id": str(patient_id),
                        "encounter_id": str(enc["encounter_id"]),
                        "text": text,
                        "label": list(set([x.split(".")[0] for x in labels])),
                    }
                )

            return sample_list


def main(config) -> None:
    if skip_build(config):
        return
    ICD10DatasetBuilder(config).prepare(split_ratio=0.8)
