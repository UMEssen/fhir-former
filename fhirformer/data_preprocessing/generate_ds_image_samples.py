import logging
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

from fhirformer.data_preprocessing.encounter_dataset_builder import (
    EncounterDatasetBuilder,
)
from fhirformer.data_preprocessing.util import (
    get_valid_labels,
    load_datastore,
    skip_build,
)

logger = logging.getLogger(__name__)


class ImageDatasetBuilder(EncounterDatasetBuilder):
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
        - Patient needs at least one imaging study and one encounter
        """
        pat_data = datastore.filter_patient(patient_id=patient_id)

        if (
            len(pat_data.patient_df) == 0
            or len(pat_data.resources["imaging_study"]) == 0
            or len(pat_data.resources["encounter"]) == 0
        ):
            return []

        patient_metadata_str = (
            f"Patient metadata:\n{self.pat_df_to_string(pat_data.patient_df)}\n\n"
        )
        sample_list = []
        for enc in pat_data.resources["encounter"].itertuples(index=False):
            # get imaging study for encounter duration

            imaging_study_during_enc = pat_data.filter_patient(
                patient_id=patient_id,
                start_filter_date=enc.start,
                end_filter_date=enc.end,
                target_resource="imaging_study",
                end_inclusive=False,
            )
            if len(imaging_study_during_enc.resources["imaging_study"]) == 0:
                continue
            else:
                # get the label
                labels = (
                    imaging_study_during_enc.resources["imaging_study"][
                        "procedure_code"
                    ]
                    .value_counts()
                    .keys()
                ).tolist()

                labels = [
                    label
                    for label in labels
                    if label in self.config["valid_procedure_codes"]
                ]
                labels = [item for sublist in labels for item in sublist]

                if len(labels) == 0:
                    continue

                # Get data before the beginning of this particular encounter
                resources_before_enc = pat_data.filter_patient(
                    patient_id=patient_id,
                    end_filter_date=enc.start,
                    end_inclusive=False,
                ).resources

                tumor_string = self.get_tumors(resources_before_enc["episode_of_care"])
                if len(tumor_string) > 0:
                    tumor_string = f"Tumor history: {tumor_string}\n\n"

                pat_hist = self.pat_history_to_string(resources_before_enc)

                if not pat_hist:
                    continue

                text = (
                    f"{patient_metadata_str}"
                    f"Encounter:\n{self.enc_to_string(enc)}\n\n"
                    f"{tumor_string}"
                    f"ICD Version: {self.get_icd_version(resources_before_enc['condition'])}\n\n"
                    f"Patient history:\n{pat_hist}"
                )

                sample_list.append(
                    {
                        "patient_id": str(patient_id),
                        "encounter_id": enc.encounter_id,
                        "encounter_start_date": str(enc.start),
                        "text": text,
                        "labels": labels,
                    }
                )
        return sample_list


def main(config):
    if skip_build(config):
        return

    config["valid_procedure_codes"] = get_valid_labels(
        config["root_dir"] / Path("ds_image/imaging_study.pkl"), "procedure_code"
    )
    ImageDatasetBuilder(config).prepare(split_ratio=0.8)
