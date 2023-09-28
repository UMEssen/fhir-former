from datetime import datetime

import pandas as pd
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


class PreTrainDatasetBuilder(EncounterDatasetBuilder):
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
            "episode_of_care",
            "diagnostic_report",
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

    @staticmethod
    def get_age_from_birth_date(birth_date: str) -> int:
        return (datetime.now() - pd.to_datetime(birth_date)).days // 365

    @staticmethod
    def dict_to_string(d):
        return "\n".join(["\t".join([str(k), str(v)]) for k, v in d.items()])

    def process_patient(self, patient_id):
        """
        - Patient needs to have at least one encounter
        - Patient history within encounter len must be > 0
        """

        pat_data = store_list_global[self.index].filter_patient(patient_id=patient_id)

        if len(pat_data.resources["encounter"]) == 0:
            return []

        patient_metadata_str = (
            f"Patient metadata:\n{self.pat_df_to_string(pat_data.patient_df)}\n\n"
        )
        sample_list = []

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

        return sample_list


def main(config) -> None:
    if skip_build(config):
        return
    PreTrainDatasetBuilder(config).prepare(split_ratio=0.8)
