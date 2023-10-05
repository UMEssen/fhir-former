import logging
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

from fhirformer.data_preprocessing.encounter_dataset_builder import (
    EncounterDatasetBuilder,
)
from fhirformer.data_preprocessing.util import load_datastore, skip_build

logger = logging.getLogger()


class ICD10DatasetBuilder(EncounterDatasetBuilder):
    def __init__(self, config):
        super().__init__(config)

    def global_multiprocessing(self):
        results = []
        for datastore_path in tqdm(
            sorted(self.ds_folder.glob("datastore*.pkl")),
            desc="Overall progress of datastores",
        ):
            global datastore
            datastore = load_datastore(datastore_path)
            with Pool(
                processes=self.num_processes,
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
        if (
            len(pat_data.resources["encounter"]) == 0
            or len(pat_data.resources["condition"]) == 0
        ):
            return sample_list

        patient_metadata_str = (
            f"Patient metadata:\n{self.pat_df_to_string(pat_data.patient_df)}\n\n"
        )

        for enc in pat_data.resources["encounter"].itertuples(index=False):
            # TODO: Why do we need this?
            # resources_during_enc = pat_data.filter_patient(
            #     patient_id=patient_id,
            #     start_filter_date=enc.start,
            #     end_filter_date=enc.end,
            # ).resources
            # if len(resources_during_enc["imaging_study"]) == 0:
            #     continue

            duration = (enc.end - enc.start).days
            if duration <= 2:
                continue

            # Generate multiple samples from each encounter
            for date in pd.date_range(
                enc.start + pd.Timedelta(days=1), enc.end - pd.Timedelta(days=1)
            ):
                # TODO: Why do we need this?
                # resources_during_sample = pat_data.filter_patient(
                #     patient_id=patient_id,
                #     start_filter_date=enc.start,
                #     end_filter_date=date,
                # ).resources
                # if len(resources_during_sample["condition"]) == 0:
                #     continue
                # From date on we take the conditions
                condition_labels_df = pat_data.filter_patient(
                    patient_id=patient_id,
                    start_filter_date=date,
                    end_filter_date=enc.end,
                    target_resource="condition",
                ).resources["condition"]

                # TODO: make sure that there we have new conditions compared to the last sample

                labels = (
                    condition_labels_df.loc[
                        ~condition_labels_df["icd_code"].str.startswith(
                            ("GB", "GROUPBOX", "PATSTURZ")
                        ),
                        "icd_code",
                    ]
                    .unique()
                    .tolist()
                )

                # Skip if no labels
                if not labels:
                    continue

                resources_until_date = pat_data.filter_patient(
                    patient_id=patient_id,
                    end_filter_date=date - pd.Timedelta(days=1),
                ).resources
                pat_hist = self.pat_history_to_string(resources_until_date)

                if not pat_hist:
                    continue

                text = (
                    f"{patient_metadata_str}"
                    f"Encounter:\n{self.enc_to_string(enc)}\n\n"
                    f"Duration: {duration}\n\n"
                    f"ICD Version: {self.get_icd_version(resources_until_date['condition'])}\n\n"
                    f"Patient journey:\n{pat_hist}"
                )

                unique_labels = list(set([x.split(".")[0] for x in labels]))
                if len(unique_labels) == 0:
                    raise ValueError(f"No labels were generated for {labels}.")
                sample_list.append(
                    {
                        "patient_id": str(patient_id),
                        "encounter_id": str(enc.id),
                        "text": text,
                        "labels": list(set([x.split(".")[0] for x in labels])),
                    }
                )

            return sample_list


def main(config) -> None:
    if skip_build(config):
        return
    ICD10DatasetBuilder(config).prepare(split_ratio=0.8)
