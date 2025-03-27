import logging
from multiprocessing import Pool
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from fhirformer.data_preprocessing.data_store import DataStore
from fhirformer.data_preprocessing.encounter_dataset_builder import (
    EncounterDatasetBuilder,
)
from fhirformer.data_preprocessing.util import load_datastore, skip_build

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
                            chunksize=5,
                        ),
                        total=len(datastore.patient_list),
                        desc="Processing patients within current datastore",
                    )
                )

            # Remove empty lists
            results_iter = [x for x in results_iter if x]
            results.append(results_iter)
            if self.config["debug"]:
                break
        return results

    @staticmethod
    def get_filtered_image_labels(labels: List[str]) -> List[str]:
        selected_labels = ["CR", "CT", "MR", "US", "XA", "NM", "OT"]
        filtered_labels = [lab if lab in selected_labels else "OT" for lab in labels]
        return filtered_labels

    def _validate_encounter_dates(
        self, enc
    ) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        """Validate and convert encounter start/end dates."""
        try:
            enc_start = pd.to_datetime(pd.Series([enc.start]))[0]
            enc_end = pd.to_datetime(pd.Series([enc.end]))[0]
            if pd.isna(enc_start) or pd.isna(enc_end):
                logger.debug(f"Skipping encounter {enc.encounter_id} - invalid dates")
                return None, None
            return enc_start, enc_end
        except Exception as e:
            logger.error(
                f"Error converting dates for encounter {enc.encounter_id}: {str(e)}"
            )
            return None, None

    def _get_patient_metadata(
        self, pat_data: DataStore, patient_id: str, enc_start_str: str
    ) -> str | None:
        """Get patient metadata string."""
        try:
            filtered_patient = pat_data.filter_patient(patient_id=patient_id)
            if filtered_patient is None:
                logger.error(f"Could not get patient data for {patient_id}")
                return None

            filtered_df = pd.DataFrame(filtered_patient.patient_df)
            if filtered_df.empty:
                logger.error(f"Empty filtered patient data for {patient_id}")
                return None

            return f"Patient metadata:\n{self.pat_df_to_string(filtered_df, enc_start_str)}\n\n"
        except Exception as e:
            logger.error(f"Error generating patient metadata: {str(e)}")
            return None

    def _process_window(
        self,
        pat_data: DataStore,
        patient_id: str,
        date: pd.Timestamp,
        enc_end: pd.Timestamp,
        previous_history: str | None,
        previous_labels: list | None,
    ) -> tuple[list | None, str | None, list | None]:
        """Process a single window and return labels, patient history, and updated previous values."""
        try:
            # Get imaging study for encounter duration
            filtered_data = pat_data.filter_patient(
                patient_id=patient_id,
                start_filter_date=date,
                end_filter_date=enc_end,
                target_resource="imaging_study",
                start_inclusive=True,
                end_inclusive=True,
            )

            if filtered_data is None or filtered_data.resources["imaging_study"].empty:
                logger.debug("No imaging studies found for window")
                return None, None, None

            # Get labels
            labels = (
                filtered_data.resources["imaging_study"]["modality_code"].unique()
            ).tolist()
            labels = self.get_filtered_image_labels(labels)

            if "nonull" in self.config["data_id"]["ds_image"] and len(labels) == 0:
                logger.debug("Skipping window - no valid labels")
                return None, None, None

            # Get patient history
            resources_before_date = pat_data.filter_patient(
                patient_id=patient_id,
                end_filter_date=date,
                end_inclusive=False,
            )

            if resources_before_date is None:
                logger.debug("No resources found before date")
                return None, None, None

            pat_hist = self.pat_history_to_string(resources_before_date.resources)

            if not pat_hist or (
                pat_hist == previous_history and labels == previous_labels
            ):
                logger.debug("Skipping window - no new information")
                return None, None, None

            if not labels:
                logger.debug("Skipping window - no labels")
                return None, None, None

            return labels, pat_hist, labels
        except Exception as e:
            logger.error(f"Error processing window: {str(e)}")
            return None, None, None

    def process_patient_sliding_window(
        self, patient_id: str, pat_data: DataStore
    ) -> List[Dict[str, Any]]:
        """Process patient data using sliding window approach."""
        if (
            "encounter" not in pat_data.resources
            or "imaging_study" not in pat_data.resources
        ):
            logger.error("Missing required resources")
            return []

        sample_list = []
        encounters_df = pat_data.resources["encounter"]

        for enc in encounters_df.itertuples(index=False):
            # Validate encounter dates
            enc_start, enc_end = self._validate_encounter_dates(enc)
            if enc_start is None or enc_end is None:
                continue

            # Check duration
            duration = (enc_end - enc_start).days
            if duration <= 2:
                logger.debug(
                    f"Skipping encounter {enc.encounter_id} - duration too short ({duration} days)"
                )
                continue

            # Get patient metadata
            enc_start_str = enc_start.strftime("%Y-%m-%d %H:%M:%S")
            patient_metadata_str = self._get_patient_metadata(
                pat_data, patient_id, enc_start_str
            )
            if patient_metadata_str is None:
                continue

            # Process windows
            previous_history = None
            previous_labels = None
            max_windows = 30

            try:
                date_range = pd.date_range(
                    enc_start + pd.Timedelta(days=1),
                    enc_end - pd.Timedelta(days=1),
                    tz="UTC",
                )
            except Exception as e:
                logger.error(f"Error creating date range: {str(e)}")
                continue

            logger.debug(
                f"Processing encounter {enc.encounter_id} with {len(date_range)} potential windows"
            )

            for count, date in enumerate(date_range):
                if count == max_windows:
                    break

                # Process window
                labels, pat_hist, new_labels = self._process_window(
                    pat_data,
                    patient_id,
                    date,
                    enc_end,
                    previous_history,
                    previous_labels,
                )
                if labels is None:
                    continue

                previous_labels = new_labels
                previous_history = pat_hist

                # Create sample
                text = (
                    f"{patient_metadata_str}"
                    f"Encounter:\n{self.enc_to_string(enc)}\n\n"
                    f"Patient history:\n{pat_hist}"
                )
                sample_list.append(
                    {
                        "patient_id": str(patient_id),
                        "encounter_id": enc.encounter_id,
                        "encounter_start": str(enc_start),
                        "sample_start": str(date),
                        "duration": duration,
                        "text": text,
                        "labels": labels,
                    }
                )
                logger.debug(f"Added sample for window {count} with labels {labels}")

        logger.debug(f"Generated {len(sample_list)} samples for patient {patient_id}")
        return sample_list

    def process_patient_one_day_after(self, patient_id: str, pat_data: pd.DataFrame):
        sample_list = []
        for enc in pat_data.resources["encounter"].itertuples(index=False):
            patient_metadata_str = f"Patient metadata:\n{self.pat_df_to_string(pat_data.patient_df, enc.start)}\n\n"
            sample_start = enc.start + pd.Timedelta(days=1)
            # Get data before the beginning of this particular encounter
            resources_before_enc = pat_data.filter_patient(
                patient_id=patient_id,
                end_filter_date=sample_start,
                end_inclusive=False,
            ).resources
            pat_hist = self.pat_history_to_string(resources_before_enc)

            if not pat_hist:
                continue

            # Get imaging study for encounter duration
            imaging_study_during_enc = pat_data.filter_patient(
                patient_id=patient_id,
                start_filter_date=sample_start,
                end_filter_date=enc.end,
                target_resource="imaging_study",
                start_inclusive=True,
                end_inclusive=False,
            )
            # Label explanation can be found here: http://dicomlookup.com/modalities.asp
            labels = (
                imaging_study_during_enc.resources["imaging_study"][
                    "modality_code"
                ].unique()
            ).tolist()
            labels = self.get_filtered_image_labels(labels)

            if "nonull" in self.config["data_id"]["ds_image"] and len(labels) == 0:
                continue

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
                    "sample_start": str(sample_start),
                    "duration": (enc.end - enc.start).days,
                    "text": text,
                    "labels": labels,
                }
            )
        return sample_list

    def process_patient(self, patient_id: str):
        # Get the patient data from datastore
        pat_data = datastore.filter_patient(patient_id=patient_id)

        # Skip patients that have no encounters or no imaging studies
        if (
            pat_data is None
            or pat_data.patient_df.empty
            or pat_data.resources["encounter"].empty
            or pat_data.resources["imaging_study"].empty
        ):
            logger.debug(f"Skipping patient {patient_id} - missing required data")
            return []

        if "nosliding" in self.config["data_id"]["ds_image"]:
            return self.process_patient_one_day_after(patient_id, pat_data)
        else:
            return self.process_patient_sliding_window(patient_id, pat_data)


def main(config):
    if skip_build(config):
        return

    if "nosliding" in config["data_id"]["ds_image"]:
        logger.info("Generating samples for ds_image without sliding window.")
    else:
        logger.info("Generating samples for ds_image using a sliding window.")

    builder = ImageDatasetBuilder(config)

    # Track statistics
    total_patients = 0
    total_encounters = 0
    total_samples = 0

    def count_stats(samples):
        nonlocal total_patients, total_encounters, total_samples
        if samples:
            total_patients += 1
            encounter_ids = set(s["encounter_id"] for s in samples)
            total_encounters += len(encounter_ids)
            total_samples += len(samples)

    # Wrap prepare to collect statistics
    results = builder.global_multiprocessing()
    for result_batch in results:
        for patient_samples in result_batch:
            count_stats(patient_samples)

    logger.info(f"Generated samples from {total_patients} patients")
    logger.info(f"Total encounters processed: {total_encounters}")
    logger.info(f"Total samples generated: {total_samples}")

    builder.prepare(split_ratio=0.8)
