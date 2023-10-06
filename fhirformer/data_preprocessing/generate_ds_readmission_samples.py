import logging
from datetime import timedelta
from multiprocessing import Pool

from tqdm import tqdm

from fhirformer.data_preprocessing.encounter_dataset_builder import (
    EncounterDatasetBuilder,
)
from fhirformer.data_preprocessing.util import load_datastore, skip_build

logger = logging.getLogger(__name__)


class ReadmissionDatasetBuilder(EncounterDatasetBuilder):
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

    def process_patient(self, patient_id: str):
        """
        - Patient needs at least one encounter
        - We check whether there are encounters that happened after this first encounter and within 30 days
        """
        pat_data = datastore.filter_patient(patient_id=patient_id)

        if len(pat_data.resources["encounter"]) == 0:
            return []

        patient_metadata_str = (
            f"Patient metadata:\n{self.pat_df_to_string(pat_data.patient_df)}\n\n"
        )
        sample_list = []
        for enc in pat_data.resources["encounter"].itertuples():
            readmission = pat_data.filter_patient(
                patient_id=patient_id,
                start_filter_date=enc.end,
                end_filter_date=enc.end + timedelta(days=30),
                target_resource="encounter",
            ).resources["encounter"]
            if len(readmission) == 0:
                labels = 0
            else:
                labels = 1
            # Get data before the end of this particular encounter
            resources_before_end = pat_data.filter_patient(
                patient_id=patient_id, end_filter_date=enc.end
            ).resources

            tumor_string = self.get_tumors(resources_before_end["episode_of_care"])
            if len(tumor_string) > 0:
                tumor_string = f"Tumor history: {tumor_string}\n\n"

            pat_hist = self.pat_history_to_string(resources_before_end)

            if not pat_hist:
                continue

            text = (
                f"{patient_metadata_str}"
                f"Encounter:\n{self.enc_to_string(enc)}\n\n"
                f"{tumor_string}"
                f"ICD Version: {self.get_icd_version(resources_before_end['condition'])}\n\n"
                f"Patient history:\n{pat_hist}"
            )

            sample_list.append(
                {
                    "patient_id": str(patient_id),
                    "encounter_id": enc.id,
                    "encounter_start_date": str(enc.start),
                    "text": text,
                    "labels": labels,
                }
            )
        return sample_list


def main(config):
    if skip_build(config):
        return

    ReadmissionDatasetBuilder(config).prepare(split_ratio=0.8)
