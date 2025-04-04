from multiprocessing import Pool

from tqdm import tqdm

from fhirformer.data_preprocessing.encounter_dataset_builder import (
    EncounterDatasetBuilder,
)
from fhirformer.data_preprocessing.util import load_datastore, skip_build


class ICD10MainDatasetBuilder(EncounterDatasetBuilder):
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
            - Duration of encounter must be at least 2 days
            - One "hauptdiagnose" must be present
            - label may not start with "???"
        Sampling:
            - One Encounter equals one sample
            - Idea: Find the main diagnosis for a encounter
        """
        pat_data = datastore.filter_patient(patient_id=patient_id)

        sample_list = []
        if (
            len(pat_data.resources["encounter"]) == 0
            or len(pat_data.resources["condition"]) == 0
        ):
            return sample_list

        for enc in pat_data.resources["encounter"].itertuples(index=False):
            patient_metadata_str = f"Patient metadata:\n{self.pat_df_to_string(pat_data.patient_df, enc.start)}\n\n"
            resources_during_enc = pat_data.filter_patient(
                patient_id=patient_id,
                start_filter_date=enc.start,
                end_filter_date=enc.end,
                end_inclusive=True,
            ).resources
            if len(resources_during_enc["imaging_study"]) == 0:
                continue

            duration = (enc.end - enc.start).days
            if duration <= 2:
                continue

            # Condition corner
            # todo code_med_hauptdiagnose is now code or sth check ;)
            # TODO: is it icd_code?
            # TODO: What resources do we use for predicting? Everything but encounter?
            raise NotImplementedError("Fix todos before running this!")
            con_label = (
                resources_during_enc["condition"]
                .loc[resources_during_enc["condition"]["icd_code"]]
                .reset_index(drop=True)
            )

            assert len(con_label) == 1, "Only one main diagnosis per encounter allowed"

            # Skip if no labels
            if not con_label:
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

            sample_list.append(
                {
                    "patient_id": str(patient_id),
                    "encounter_id": enc.encounter_id,
                    "text": text,
                    "label": con_label.tolist(),
                }
            )

        return sample_list


def main(config) -> None:
    if skip_build(config):
        return
    ICD10MainDatasetBuilder(config).prepare(split_ratio=0.8)
