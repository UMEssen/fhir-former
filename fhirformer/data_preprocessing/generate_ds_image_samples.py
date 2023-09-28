import logging
from pathlib import Path

from fhirformer.data_preprocessing.encounter_dataset_builder import (
    EncounterDatasetBuilder,
)
from fhirformer.data_preprocessing.util import (
    get_column_map_txt_resources,
    get_data_info,
    get_patient_ids_lists,
    get_valid_labels,
    skip_build,
    validate_resources,
)

logger = logging.getLogger(__name__)


class ImageDatasetBuilder(EncounterDatasetBuilder):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
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
        self.filtered_column_map_txt_resources = get_column_map_txt_resources(
            config, resources_for_pre_training
        )

        # this has proven to be the fastest way to process patients so far
        global store_list_global
        store_list_global = self.get_source_data(
            split_data=False,
            n=75,
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
        - Patient needs at least one imaging study and one encounter
        """
        pat_data = store_list_global[self.index].filter_patient(patient_id=pat)

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
        for _, enc in pat_data.resources["encounter"].iterrows():
            # get imaging study for encounter duration

            imaging_study_during_enc = pat_data.filter_patient(
                patient_id=pat,
                start_filter_date=enc["start"],
                end_filter_date=enc["end"],
                target_resource="imaging_study",
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

                # get all fucking data before the encounter starts
                resources_before_enc = pat_data.filter_patient(
                    patient_id=pat, end_filter_date=enc["start"]
                ).resources

                pat_hist = self.pat_history_to_string(resources_before_enc)

                if not pat_hist:
                    continue

                text = (
                    f"{patient_metadata_str}"
                    f"Encounter:\n{self.enc_to_string(enc)}\n\n"
                    f"ICD Version: {self.get_icd_version(resources_before_enc['condition'])}\n\n"
                    f"Patient history:\n{pat_hist}"
                )

                sample_list.append(
                    {
                        "patient_id": str(pat),
                        "encounter_id": enc["id"],
                        "encounter_start_date": str(enc["start"]),
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
