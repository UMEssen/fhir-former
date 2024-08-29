import logging
from datetime import timedelta
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

from fhirformer.data_preprocessing.encounter_dataset_builder import (
    EncounterDatasetBuilder,
)
from fhirformer.data_preprocessing.util import load_datastore, skip_build

logger = logging.getLogger(__name__)


class MortalityRiskDatasetBuilder(EncounterDatasetBuilder):
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
        - Input: Historical data of patient before encounter + 2 days
        - Target: did patient die during encounter?
        """
        pat_data = datastore.filter_patient(patient_id=patient_id)

        if (
            len(pat_data.resources["encounter"]) == 0
        ):  # todo think about only taking patients that died at some point
            return []

        sample_list = []
        for enc in pat_data.resources["encounter"].itertuples(index=False):
            patient_metadata_str = f"Patient metadata:\n{self.pat_df_to_string(pat_data.patient_df, enc.start)}\n\n"
            sample_start = enc.start + pd.Timedelta(days=2)
            # Get data before the beginning of this particular encounter
            resources_before_enc = pat_data.filter_patient(
                patient_id=patient_id,
                end_filter_date=sample_start,
                end_inclusive=False,
            ).resources
            pat_hist = self.pat_history_to_string(resources_before_enc)

            if not pat_hist:
                continue

            assert (
                len(pat_data.patient_df.deceased_date) == 1
            ), "Patient should have only one deceased date"

            if not pat_data.patient_df.deceased_date.empty:
                deceased_date = pat_data.patient_df.deceased_date.iloc[0]
            else:
                deceased_date = None

            # did patient die during encounter?
            if deceased_date:
                # in some cases the time of death is set to one day after the encounter
                decesased_during_encounter = (
                    deceased_date >= enc.start
                    and deceased_date <= enc.end + timedelta(days=1)
                )

            text = (
                f"{patient_metadata_str}"
                f"Encounter:\n{self.enc_to_string(enc)}\n\n"
                f"Patient history:\n{pat_hist}"
            )

            sample_list.append(
                {
                    "patient_id": str(patient_id),
                    "encounter_id": enc.encounter_id,
                    "encounter_start": str(enc.start),
                    "encounter_end": str(enc.end),
                    "sample_start": str(sample_start),
                    "duration": (enc.end - enc.start).days,
                    "text": text,
                    "labels": decesased_during_encounter,
                }
            )
        return sample_list


def main(config):
    if skip_build(config):
        return

    MortalityRiskDatasetBuilder(config).prepare(split_ratio=0.8)
