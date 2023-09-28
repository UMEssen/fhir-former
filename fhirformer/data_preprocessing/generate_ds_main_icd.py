from fhirformer.data_preprocessing.encounter_dataset_builder import (
    EncounterDatasetBuilder,
)
from fhirformer.data_preprocessing.util import (
    get_column_map_txt_resources,
    get_data_info,
    get_patient_ids_lists,
    validate_resources,
    skip_build,
)


class ICD10MainDatasetBuilder(EncounterDatasetBuilder):
    def __init__(self, config):
        # Here you can simply decide which resources you want to use for the pre-training
        resources_for_pre_training = [
            "procedure",
            "condition",
            "imaging_study",
            "biologically_derived_product",
            "observation",
            "service_request",
            "medication",
            # episode_of_care
            # diagnostic_report
        ]

        super().__init__(config)
        self.config = config

        self.filtered_column_map_txt_resources = get_column_map_txt_resources(
            config, resources_for_pre_training
        )

        # this has proven to be the fastest way to process patients so far
        global store_list_global
        store_list_global = self.get_source_data(
            split_data=True,
            n=2,
        )

        # pass the fucking patient ids in a list each
        self.patient_ids_lists = get_patient_ids_lists(store_list_global)

        # get some info
        pats_int = sum([len(x) for x in self.patient_ids_lists])
        get_data_info(pats_int, store_list_global)
        self.index = None

        # filter column_maps by resources_with_date_column
        validate_resources(resources_for_pre_training, self.config)

    def process_patient(self, pat):
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
        store = store_list_global
        pat_data = store.filter_patient(patient_id=pat)

        sample_list = []
        if len(pat_data.enc) == 0 or len(pat_data.con) == 0:
            return []

        patient_metadata_str = (
            f"Patient metadata:\n{self.pat_df_to_string(pat_data.patient_df)}\n\n"
        )

        for _, enc in pat_data.enc.iterrows():
            resources_during_enc = pat_data.filter_patient(
                patient_id=pat,
                start_filter_date=enc["start"],
                end_filter_date=enc["end"],
            )
            if len(resources_during_enc["imaging_study"]) == 0:
                continue

            duration = (enc["end"] - enc["start"]).days
            if duration <= 2:
                continue

            resources_during_enc["condition"]

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
                    "patient_id": str(pat),
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
