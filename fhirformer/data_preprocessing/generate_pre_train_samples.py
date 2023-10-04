from fhirformer.data_preprocessing.encounter_dataset_builder import (
    EncounterDatasetBuilder,
)
from fhirformer.data_preprocessing.util import skip_build, load_datastore
import logging
from typing import Dict, List
from tqdm import tqdm
from multiprocessing import Pool

logger = logging.getLogger(__name__)


class PreTrainDatasetBuilder(EncounterDatasetBuilder):
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
                processes=self.NUM_PROCESSES,
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

    def process_patient(self, patient_id: str) -> List[Dict]:
        """
        - Patient needs to have at least one encounter
        - Patient history within encounter len must be > 0
        """
        pat_data = datastore.filter_patient(patient_id=patient_id)

        tumor_string = self.get_tumors(pat_data.resources["episode_of_care"])

        if len(tumor_string) > 0:
            tumor_string = f"Tumor history: {tumor_string}\n\n"

        if len(pat_data.resources["encounter"]) == 0:
            return []

        patient_metadata_str = (
            f"Patient metadata:\n{self.pat_df_to_string(pat_data.patient_df)}\n\n"
        )
        sample_list = []

        logger.debug(
            f"Patient {patient_id} has {len(pat_data.resources['encounter'])} encounters"
        )
        for enc in pat_data.resources["encounter"].itertuples(index=False):
            duration = (enc.end - enc.start).days

            resources_during_enc = pat_data.filter_patient(
                patient_id=patient_id,
                start_filter_date=enc.start,
                end_filter_date=enc.end,
            ).resources

            pat_hist = self.pat_history_to_string(resources_during_enc)

            if not pat_hist:
                continue

            text = (
                f"{patient_metadata_str}"
                f"Encounter:\n{self.enc_to_string(enc)}\n\n"
                f"Duration: {duration}\n\n"
                f"{tumor_string}"
                f"ICD Version: {self.get_icd_version(resources_during_enc['condition'])}\n\n"
                f"Patient journey in stay:\n{pat_hist}"
            )

            sample_list.append(
                {
                    "patient_id": str(patient_id),
                    "encounter_id": str(enc.id),
                    "text": text,
                }
            )

        logger.debug(f"Patient {patient_id} has {len(sample_list)} samples")
        return sample_list


def main(config) -> None:
    if skip_build(config):
        return
    PreTrainDatasetBuilder(config).prepare(split_ratio=0.8)
