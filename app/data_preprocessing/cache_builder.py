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
    patient_path_parents = Path(config["patient_parent_path_filtered"])

    # METRICS
    observation_path = Path(config["observation_path"])
    observation_path_filtered = Path(config["observation_path_filtered"])

    extract = FHIRExtractor(config)
    filter = FHIRFilter(config)
    validator = FHIRValidator(config)

    # Filters
    # Only stationary patients
    # Encounters must be at least 2 day long
    # Trash 50 % of all patients as they shall still be used in other studies or downstream tasks

    # 1. Encounter -> Take only IMG (impatient encounters), 2010-2022  -> 2021-2022 for test
    # 2. Condition -> Take ENC cohort -> Take only pats which have at least x conditions
    # 3. Procedure -> Pats from enc
    # 4. Patient -> Pats from enc

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

    # find patient parent ids
    if not patient_path_parents.exists() or config["reload_cache"]:
        logging.info(f"Finding patient metas")
        extract.build_filter_patient_parents()

    # Condition -> code, display, recordedDate
    if not condition_path.exists() or config["reload_cache"]:
        logging.info(f"Extracting Condition Data")
        extract.build_condition()

    if not condition_path_filtered.exists() or config["reload_cache"]:
        logging.info("Filtering relevant conditions")
        filter.filter_conditions()
        logging.info("Validating conditions")
        validator.validate_conditions()
        exit(0)

    # Procedure -> Procedure Code, Procedure Root Code, Practitioner, performedDateTime, cat -> med -> Hauptdiagnose
    if not procedure_path.exists() or config["reload_cache"]:
        logging.info(f"Extracting Procedure...")
        extract.build_procedure()

    if not procedure_path_filtered.exists() or config["reload_cache"]:
        logging.info(f"Filtering Procedure...")
        filter.filter_procedures()
        logging.info("Validating Procedure...")
        validator.validate_procedures()

    if not observation_path.exists() or config["reload_cache"]:
        logging.info(f"Extracting Observation...")
        extract.build_observation()
        # TODO add filter and validation, obs needs Reports + Tumor Docus
