import logging

import pandas as pd

from fhirformer.fhir.util import OUTPUT_FORMAT, check_and_read, col_to_datetime

logger = logging.getLogger(__name__)


class FHIRValidator:
    def __init__(self, config):
        self.config = config
        self.resource_schemas = {
            "condition": [("patient_id", True), ("icd_code", True)],
            "encounter": [("patient_id", True)],
            "biologically_derived_product": [("ausgabe_datetime", False)],
            "patient": [("patient_id", True)],
            "procedure": [("patient_id", False)],
            "medicationstatement": [
                ("medicationName", True),
                ("patient_id", True),
                ("event_time", True),
            ],
            "observation": [("patient_id", True), ("value_quantity", True)],
            "imaging_study": [("patient_id", True)],
            "episode_of_care": [("patient_id", True)],
            "service_request": [("patient_id", True)],
            "diagnostic_report": [("patient_id", True)],
            "medication": [("patient_id", True)],
        }

    def validate(self, resource: str):
        # TODO: Add check for unique IDs for each resource type
        resource = resource.lower()
        if self.config["skip_validation"]:
            pass
        elif resource in self.resource_schemas:
            self.generic_validate(resource, self.resource_schemas[resource])
        else:
            raise NotImplementedError(f"Resource {resource} not supported")

    def generic_validate(self, resource, schema):
        df = check_and_read(self.config["task_dir"] / f"{resource}{OUTPUT_FORMAT}")

        # Skip validation if DataFrame is empty
        if df.empty:
            logger.warning(f"Skipping validation for {resource}: DataFrame is empty")
            return

        if resource == "biologically_derived_product":
            df["ausgabe_datetime"] = col_to_datetime(df.ausgabe_datetime)
            df_count = df.ausgabe_datetime.value_counts().sort_index()[:-1]
            if (df_count == 0).any():
                logger.warning("BDP count for one or more imported days = 0")

        na_counts = df.isna().sum()

        for field_name, is_error in schema:
            # Skip validation if required column doesn't exist
            if field_name not in df.columns:
                logger.warning(
                    f"Skipping validation for {field_name} in {resource}: Column does not exist"
                )
                continue
            self.na_checker(field_name, na_counts, is_error)

    @staticmethod
    def na_checker(field_name: str, na_counts: pd.Series, is_error: bool) -> None:
        if na_counts[field_name] and is_error:
            logger.error(f"At least one {field_name} is zero")
            raise ValueError(f"At least one {field_name} is zero")
        elif na_counts[field_name]:
            logger.warning(f"At least one {field_name} is zero")
        else:
            logger.info(f"Validation for {field_name} passed")
