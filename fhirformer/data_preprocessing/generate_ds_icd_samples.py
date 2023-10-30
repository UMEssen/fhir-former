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

        for enc in pat_data.resources["encounter"].itertuples(index=False):
            duration = (enc.end - enc.start).days
            if duration <= 2:
                continue

            patient_metadata_str = f"Patient metadata:\n{self.pat_df_to_string(pat_data.patient_df, enc.start)}\n\n"
            previous_history = None
            # Generate multiple samples from each encounter
            for date in pd.date_range(
                enc.start + pd.Timedelta(days=1), enc.end - pd.Timedelta(days=1)
            ):
                # From date on we take the conditions
                condition_labels_df = pat_data.filter_patient(
                    patient_id=patient_id,
                    start_filter_date=date,
                    end_filter_date=enc.end,
                    target_resource="condition",
                    start_inclusive=True,
                    end_inclusive=True,
                ).resources["condition"]

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

                unique_labels = sorted(set([x.split(".")[0] for x in labels]))
                if len(unique_labels) == 0:
                    raise ValueError(f"No labels were generated for {labels}.")

                resources_until_date = pat_data.filter_patient(
                    patient_id=patient_id,
                    end_filter_date=date,
                    end_inclusive=False,
                ).resources
                pat_hist = self.pat_history_to_string(resources_until_date)
                # Skip if there are no infos or if the infos are the same as the previous
                if not pat_hist or pat_hist == previous_history:
                    continue
                previous_history = pat_hist
                text = (
                    f"{patient_metadata_str}"
                    f"Encounter:\n{self.enc_to_string(enc)}\n\n"
                    f"Duration: {duration}\n"
                    f"Sample date: {date}\n\n"
                    f"ICD Version: {self.get_icd_version(resources_until_date['condition'])}\n\n"
                    f"Patient journey:\n{pat_hist}"
                )

                sample_list.append(
                    {
                        "patient_id": str(patient_id),
                        "encounter_id": enc.encounter_id,
                        "text": text,
                        "labels": unique_labels,
                    }
                )

            return sample_list


def main(config) -> None:
    if skip_build(config):
        return
    ICD10DatasetBuilder(config).prepare(split_ratio=0.8)
