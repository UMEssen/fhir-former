from fhirformer.data_preprocessing.encounter_dataset_builder import (
    EncounterDatasetBuilder,
)
from fhirformer.data_preprocessing.util import (
    skip_build,
)


class ICD10MainDatasetBuilder(EncounterDatasetBuilder):
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
        global store_list_global
        store_list_global = self.get_source_data(num_splits=2)
        self.set_up(store_list_global)

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
        pat_data = store_list_global[self.index].filter_patient(patient_id=patient_id)

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

            # Condition corner
            # todo code_med_hauptdiagnose is now code or sth check ;)
            con_label = (
                resources_during_enc["condition"]
                .loc[resources_during_enc["condition"]["code_med_hauptdiagnose"]]
                .reset_index(drop=True)
            )

            assert len(con_label) == 1, "Only one hauptdiagnose per encounter allowed"

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
                    "encounter_id": str(enc["encounter_id"]),
                    "text": text,
                    "label": con_label.tolist(),
                }
            )

        return sample_list


def main(config) -> None:
    if skip_build(config):
        return
    ICD10MainDatasetBuilder(config).prepare(split_ratio=0.8)
