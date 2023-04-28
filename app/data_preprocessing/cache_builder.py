import logging
import datetime

from app.data_preprocessing.extract_transform_validate import (
    FHIRExtractor,
    FHIRFilter,
    FHIRValidator,
)

from pathlib import Path


# Main
def main(config):
    encounter_path = Path(config["encounter_path"])
    encounter_path_filtered = Path(config["encounter_path_filtered"])
    patient_path = Path(config["patient_path"])
    patient_path_filtered = Path(config["patient_path_filtered"])
    procedure_path = Path(config["procedure_path"])
    procedure_path_filtered = Path(config["procedure_path_filtered"])
    condition_path = Path(config["condition_path"])
    condition_path_filtered = Path(config["condition_path_filtered"])

    extract = FHIRExtractor(config)
    filter = FHIRFilter(config)
    validator = FHIRValidator(config)

    # Condition -> code, display, recordedDate
    if not condition_path.exists() or config["reload_cache"]:
        logging.info(f"Extracting Condition Data")
        extract.build_condition()

    if not condition_path_filtered.exists() or config["reload_cache"]:
        logging.info("Filtering relevant conditions")
        filter.filter_conditions()
        logging.info("Validating conditions")
        validator.validate_conditions()

    # Encounter; Case -> Start, End, Department
    if not encounter_path.exists() or config["reload_cache"]:
        logging.info(f"Extracting Encounter...")
        extract.build_encounter()

    if not encounter_path_filtered.exists() or config["reload_cache"]:
        logging.info(f"Filtering Encounter...")
        filter.filter_encounter()
        logging.info("Validating Encounter...")
        validator.validate_encounters()

    # Patient -> Sex, Age
    if not patient_path.exists() or config["reload_cache"]:
        logging.info(f"Extracting Patient...")
        extract.build_patient()

    if not patient_path_filtered.exists() or config["reload_cache"]:
        logging.info(f"Filtering Patient...")
        filter.filter_patient()
        logging.info("Validating Patient...")
        validator.validate_patient()

    # Procedure -> Procedure Code, Procedure Root Code, Practitioner, performedDateTime, cat -> med -> Hauptdiagnose
    if not procedure_path.exists() or config["reload_cache"]:
        logging.info(f"Extracting Procedure...")
        extract.build_procedure()

    if not procedure_path_filtered.exists() or config["reload_cache"]:
        logging.info(f"Filtering Procedure...")
        filter.filter_procedures()
        logging.info("Validating Procedure...")
        validator.validate_procedures()
