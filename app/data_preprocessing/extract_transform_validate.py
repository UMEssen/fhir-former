import os
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from fhir_pyrate.ahoy import Ahoy
from fhir_pyrate.pirate import Pirate
import datetime
from requests.adapters import Retry

from tqdm import tqdm

import logging
from sqlalchemy import create_engine, text


# REQUIRED PATHS
SEARCH_URL = os.environ["SEARCH_URL"]
BASIC_AUTH = os.environ["BASIC_AUTH"]
REFRESH_AUTH = os.environ["REFRESH_AUTH"]
FHIR_USER = os.environ["FHIR_USER"]
FHIR_PASSWORD = os.environ["FHIR_PASSWORD"]
METRICS_USER = os.environ["METRICS_USER"]
METRICS_PASSWORD = os.environ["METRICS_PASSWORD"]
METRICS_HOSTNAME = os.environ["METRICS_HOSTNAME"]
METRICS_PORT = os.environ["METRICS_PORT"]
METRICS_DB = os.environ["METRICS_DB"]


engine = create_engine(
    f"postgresql://{METRICS_USER}:{METRICS_PASSWORD}@{METRICS_HOSTNAME}:{METRICS_PORT}/{METRICS_DB}"
)


auth = Ahoy(
    auth_method="env",
    username=FHIR_USER,
    auth_url=BASIC_AUTH,
    refresh_url=REFRESH_AUTH,
)


with open("app/config/constants.yaml", "r") as stream:
    code_dict = yaml.safe_load(stream)


def col_to_datetime(date_series: pd.Series) -> pd.Series:
    # some procedure dates are just YYYY-MM-DD and will result in nan values => I don't fucking care right now
    if date_series.any():
        date_series = pd.to_datetime(
            date_series, format="%Y-%m-%dT%H:%M:%S.%f%z", utc=True, errors="coerce"
        ).dt.tz_convert("CET")

        if date_series is pd.NaT:
            date_series = pd.to_datetime(
                date_series, format="%Y-%m-%d %H:%M:%S.%f%z", utc=True, errors="coerce"
            ).dt.tz_convert("CET")

        date_series = date_series.dt.tz_localize(None)
    return date_series


# Class FHIRExtractor
def transform_dict_constants():
    with (pathlib.Path(__file__).parent.parent / "config" / "constants.yaml").open(
        "r"
    ) as stream:
        code_dict = yaml.safe_load(stream)

    # adding medication key to values
    for key, value in code_dict["MEDICATION_DICT_LIST"].items():
        code_dict["MEDICATION_DICT_LIST"][key].append(key)

    # translating categorry_dict_list into ICD, OPS and medication names
    dict_combo = (
        code_dict["CONDITION_DICT_LIST"]
        | code_dict["PROCEDURE_DICT_LIST"]
        | code_dict["MEDICATION_DICT_LIST"]
    )
    for key, value in code_dict["CATEGORY_DICT_LIST"].items():
        codes_in_class = list()
        for item in value:
            codes_in_class.append(dict_combo[item])
        codes_in_class = [item for sublist in codes_in_class for item in sublist]
        code_dict["CATEGORY_DICT_LIST"][key] = codes_in_class
    return code_dict


class FHIRExtractor:
    """
    desc

    Method:
        -
    Run:

    """

    def __init__(self, config):
        self.config = config
        # Pirate
        if config["read_from_fhir_cache"]:
            self.search = Pirate(
                auth=auth,
                base_url=SEARCH_URL,
                num_processes=40,
                cache_folder=self.config["bundle_cache_folder_path"],
                retry_requests=Retry(
                    total=3,  # Retries for a total of three times
                    backoff_factor=0.5,  # A backoff factor to apply between attempts, such that the requests are not run directly one after the other
                    status_forcelist=[
                        500,
                        502,
                        503,
                        504,
                    ],  # HTTP status codes that we should force a retry on
                    allowed_methods=[
                        "GET"
                    ],  # Set of uppercased HTTP method verbs that we should retry on
                ),
            )
        else:
            self.search = Pirate(
                auth=auth,
                base_url=SEARCH_URL,
                num_processes=40,
                retry_requests=Retry(
                    total=3,  # Retries for a total of three times
                    backoff_factor=0.5,
                    # A backoff factor to apply between attempts, such that the requests are not run directly one after the other
                    status_forcelist=[
                        500,
                        502,
                        503,
                        504,
                        404,
                    ],  # HTTP status codes that we should force a retry on
                    allowed_methods=[
                        "GET"
                    ],  # Set of uppercased HTTP method verbs that we should retry on
                ),
            )

    # Encounter
    @staticmethod
    def extract_encounter(bundle):
        records = []
        for entry in bundle.entry or []:
            resource = entry.resource
            resource_id = resource.id
            patient_id = resource.subject.reference.split("Patient/")[-1]
            status = resource.status
            try:
                encounter_start = resource.period.start
            except Exception as e:
                encounter_start = pd.NaT
            try:
                encounter_end = resource.period.end
            except Exception as e:
                encounter_end = pd.NaT

            try:
                if "Case" in resource.meta.extension[0].valueString:
                    encounter_type = "case"
                elif "Stay" in resource.meta.extension[0].valueString:
                    encounter_type = "stay"
                else:
                    encounter_type = np.nan
            except:
                encounter_type = np.nan

            elements = {
                "resource_id": resource_id,
                "patient_id": patient_id,
                "status": status,
                "encounter_start": encounter_start,
                "encounter_end": encounter_end,
                "encounter_type": encounter_type,
            }
            records.append(elements)

        return records

    def build_patient(self):
        encs = pd.read_feather(self.config["encounter_path_filtered"])
        encs = encs.drop_duplicates(subset=["patient_id"], keep="first")

        query_all = text(
            """SELECT "birthDate", gender, id as patient_id, code as insurance_type, _json
        From patient
        Left OUTER JOIN patient_identifier_type_coding ON patient._id = patient_identifier_type_coding._resource
        """
        )

        with engine.connect() as connection:
            df_all = pd.read_sql_query(query_all, connection)

        df_all["last_updated"] = [x["meta"]["lastUpdated"] for x in df_all["_json"]]

        # Kill duplicates and keep the last updated
        df_all["last_updated"] = pd.to_datetime(df_all["last_updated"])
        df_all = df_all.sort_values(["patient_id", "last_updated"])
        df_all = df_all.drop_duplicates(subset="patient_id", keep="last")

        merged_df = pd.merge(df_all, encs, on="patient_id", how="inner")

        merged_df["birthDate"] = [
            x if x is None else x.lower for x in merged_df["birthDate"]
        ]

        merged_df.reset_index(drop=True, inplace=False).to_feather(
            self.config["patient_path"]
        )

    # Building Encounter DataFrame
    def build_encounter(self):
        df = self.search.sail_through_search_space_to_dataframe(
            resource_type="Encounter",
            request_params=self.config["encounter_params"],
            time_attribute_name="date",
            date_init=self.config["start_datetime"],
            date_end=self.config["end_datetime"],
        )

        df.reset_index(drop=True, inplace=False).to_feather(
            self.config["encounter_path"]
        )

    # BDP
    @staticmethod
    def extract_bdp(bundle):
        records = []
        for entry in bundle.entry or []:
            resource = entry.resource
            # ResourceType: BDP
            if resource.resourceType == "BiologicallyDerivedProduct":
                try:
                    resource_id = resource.id
                except Exception as e:
                    resource_id = np.nan

                try:
                    request_id = resource.request[0].reference.split("ServiceRequest/")[
                        -1
                    ]
                except Exception as e:
                    request_id = np.nan

                try:
                    ausgabe_datetime = resource.storage[0].duration.end
                except Exception as e:
                    ausgabe_datetime = pd.NaT

                try:
                    extensions = resource.extension
                    output_to = next(
                        (
                            e.valueString
                            for e in extensions
                            if e.url
                            == "https://uk-essen.de/LAB/Nexus/Swisslab/KONSERVE/AUSGABEAN"
                        ),
                        None,
                    )
                except Exception as e:
                    extensions = np.nan
                    output_to = np.nan

                try:
                    ausgabe_type = next(
                        (
                            e.valueString
                            for e in extensions
                            if e.url
                            == "https://uk-essen.de/LAB/Nexus/Swisslab/KONSERVE/VERBRAUCH"
                        ),
                        None,
                    )
                except Exception as e:
                    ausgabe_type = np.nan

                try:
                    product_code = resource.productCode.coding
                    code = next(
                        (
                            e.code
                            for e in product_code
                            if e.system
                            == "https://uk-essen.de/LAB/Nexus/Swisslab/KONSERVE/KONSART"
                        ),
                        None,
                    )
                except Exception as e:
                    code = np.nan

                elements = {
                    "resource_type": "bdp",
                    "resource_id": resource_id,
                    "request_id": request_id,
                    "ausgabe_datetime": ausgabe_datetime,
                    "ausgabe_type": ausgabe_type,
                    "code": code,
                    "output_to": output_to,
                }
                records.append(elements)

            # ResourceType: Service Request
            if resource.resourceType == "ServiceRequest":
                request_id = resource.id
                patient_id = resource.subject.reference.split("Patient/")[-1]
                try:
                    output_to_einskurz = resource.requester.extension[0].valueString
                    output_to_einscode = resource.requester.extension[1].valueString
                except Exception as e:
                    output_to_einskurz = None
                    output_to_einscode = None

                elements = {
                    "resource_type": "sr",
                    "request_id": request_id,
                    "patient_id": patient_id,
                    "output_to_einskurz": output_to_einskurz,
                    "output_to_einscode": output_to_einscode,
                }
                records.append(elements)

        return records

    # Build BDP DataFrame
    def build_bdp(self):
        df = self.search.sail_through_search_space_to_dataframe(
            process_function=self.extract_bdp,
            resource_type="BiologicallyDerivedProduct",
            request_params=self.config["bdp_params"],
            time_attribute_name="shipStorageEnd",
            date_init=self.config["start_datetime"],
            date_end=datetime.datetime.now().date(),
        )

        bdp = df[df["resource_type"] == "bdp"]
        sr = df[df["resource_type"] == "sr"]

        merged = bdp.merge(sr, on="request_id", how="left")
        merged.drop(
            [
                "resource_type_x",
                "resource_type_y",
                "patient_id_x",
                "output_to_einskurz_x",
                "output_to_einscode_x",
                "resource_id_y",
                "ausgabe_datetime_y",
                "ausgabe_type_y",
                "code_y",
                "output_to_y",
            ],
            axis=1,
            inplace=True,
        )
        merged.columns = [
            "resource_id",
            "request_id",
            "ausgabe_datetime",
            "ausgabe_type",
            "code",
            "output_to",
            "patient_id",
            "output_to_einskurz",
            "output_to_einscode",
        ]

        merged.drop_duplicates(inplace=True)
        merged.reset_index(drop=True, inplace=False).to_feather(self.config["bdp_path"])

    # Observations
    @staticmethod
    def extract_observations(bundle):
        records = []
        for entry in bundle.entry or []:
            resource = entry.resource
            if resource.resourceType == "Observation":
                patient_id = resource.subject.reference.split("Patient/")[-1]

                try:
                    request_id = resource.basedOn[0].reference.split("ServiceRequest/")[
                        -1
                    ]
                except Exception as e:
                    request_id = np.nan

                code = resource.code.coding[0].code

                try:
                    value = resource.valueQuantity.value
                except Exception as e:
                    value = np.nan

                try:
                    unit = resource.valueQuantity.unit
                except Exception as e:
                    unit = np.nan

                # observed date
                if resource.effectiveDateTime is not None:
                    observation_date = resource.effectiveDateTime
                elif resource.issued is not None:
                    observation_date = resource.issued
                else:
                    observation_date = pd.NaT

                elements = {
                    "resource_type": "obs",
                    "patient_id": patient_id,
                    "request_id": request_id,
                    "code": code,
                    "value": value,
                    "unit": unit,
                    "observation_date": observation_date,
                }
                records.append(elements)

            if resource.resourceType == "ServiceRequest":
                request_id = resource.id
                req = resource.requester.extension
                if req is not None:
                    einskurz_from_dep = next(
                        (
                            e.valueString
                            for e in req
                            if e.url
                            == "https://uk-essen.de/LAB/Nexus/Swisslab/EINSENDER/EINSKURZ"
                        ),
                        None,
                    )
                    einscode_from_dep = next(
                        (
                            e.valueString
                            for e in req
                            if e.url
                            == "https://uk-essen.de/LAB/Nexus/Swisslab/EINSENDER/EINSCODE"
                        ),
                        None,
                    )
                else:
                    einskurz_from_dep = np.nan
                    einscode_from_dep = np.nan
                elements = {
                    "resource_type": "sr",
                    "request_id": request_id,
                    "einskurz_from_dep": einskurz_from_dep,
                    "einscode_from_dep": einscode_from_dep,
                }
                records.append(elements)

        return records

    def build_obs(self):
        df = self.search.sail_through_search_space_to_dataframe(
            process_function=self.extract_observations,
            resource_type="Observation",
            request_params=self.config["obs_params"],
            time_attribute_name="date",
            date_init=self.config["start_datetime"],
            date_end=datetime.datetime.now().date(),
        )

        obs_obs = df[df.resource_type == "obs"]
        obs_sr = df[df.resource_type == "sr"]

        obs_merged = obs_obs.merge(obs_sr, on="request_id", how="left")
        obs_merged.dropna(how="all", inplace=True)
        obs_merged.dropna(subset=["value_x"], inplace=True)
        obs_merged = obs_merged[
            [
                "patient_id_x",
                "request_id",
                "value_x",
                "observation_date_x",
                "einscode_from_dep_y",
            ]
        ]

        obs_merged.columns = [
            "patient_id",
            "request_id",
            "value",
            "observation_date",
            "einscode_from_dep",
        ]

        obs_merged.reset_index(drop=True, inplace=False).to_feather(
            self.config["obs_path"]
        )

    # Procedure
    @staticmethod
    def extract_procedure(bundle):
        records = []
        for entry in bundle.entry or []:
            resource = entry.resource
            patient_id = resource.subject.reference.split("Patient/")[-1]
            procedure_id = resource.id
            status = resource.status
            try:
                encounter = resource.encounter.reference.split("Encounter/")[-1]
            except Exception as e:
                encounter = np.nan
            try:
                code = resource.code.coding[0].code
            except Exception as e:
                code = np.nan
            try:
                code_display = resource.code.coding[0].display
            except Exception as e:
                code_display = np.nan
            if resource.performedPeriod is not None:
                procedure_start = resource.performedPeriod.start
            elif resource.performedDateTime is not None:
                procedure_start = resource.performedDateTime
            else:
                procedure_start = pd.NaT
            try:
                procedure_end = resource.performedPeriod.end
            except Exception as e:
                procedure_end = pd.NaT

            elements = {
                "patient_id": patient_id,
                "procedure_id": procedure_id,
                "encounter_id": encounter,
                "status": status,
                "code": code,
                "code_display": code_display,
                "procedure_start": procedure_start,
                "procedure_end": procedure_end,
            }
            records.append(elements)

        return records

    def build_procedure(self):
        df = self.search.sail_through_search_space_to_dataframe(
            process_function=self.extract_procedure,
            resource_type="Procedure",
            request_params=self.config["patient_params"],
            time_attribute_name="date",
            date_init=self.config["start_datetime"],
            date_end=datetime.datetime.now().date(),
        )

        df.procedure_start = df.procedure_start.astype(str)
        df.procedure_end = df.procedure_end.astype(str)
        df.reset_index(drop=True).to_feather(self.config["procedure_path"])

    # Building Condition DataFrame
    def build_condition(self):
        df = self.search.sail_through_search_space_to_dataframe(
            resource_type="Condition",
            request_params=self.config["condition_params"],
            time_attribute_name="recorded-date",
            date_init=self.config["start_datetime"],
            date_end=self.config["end_datetime"],
        )

        df.reset_index(drop=True, inplace=False).to_feather(
            self.config["condition_path"]
        )

    # Medication
    @staticmethod
    def extract_medication(bundle):
        records = []
        for entry in bundle.entry or []:
            resource = entry.resource
            importer = next(
                (
                    e.valueString
                    for e in resource.meta.extension
                    if e.url == "http://uk-essen.de/fhir/extension-importer-name"
                ),
                None,
            )
            try:
                if "cato" in importer:
                    medicationName = resource.code.text
                    source = "cato"
                elif "medico" in importer:
                    source = "medico"
                    medicationName = resource.code.coding[0].display
                else:
                    source = "other"
                    medicationName = (
                        resource.code.coding[0].display
                        if resource.code.coding[0].display
                        else np.nan
                    )
            except Exception as e:
                medicationName = resource.code.text
                source = np.nan

            lastUpdated = resource.meta.lastUpdated
            medication_id = resource.id

            elements = {
                "medicationName": medicationName,
                "source": source,
                "lastUpdated": lastUpdated,
                "medication_id": medication_id,
            }

            records.append(elements)

        return records

    @staticmethod
    def extract_medication_request(bundle):
        records = []
        for entry in bundle.entry or []:
            resource = entry.resource
            try:
                importer = next(
                    (
                        e.valueString
                        for e in resource.meta.extension
                        if e.url == "http://uk-essen.de/fhir/extension-importer-name"
                    ),
                    None,
                )
                if "cato" in importer:
                    source = "cato"
                elif "medico" in importer:
                    source = "medico"
                else:
                    source = np.nan
            except Exception:
                source = np.nan

            status = resource.status
            try:
                medication_id = resource.medicationReference.reference.split(
                    "Medication/"
                )[-1]
            except:
                medication_id = np.nan
            patient_id = resource.subject.reference.split("Patient/")[-1]
            try:
                event_time = (
                    resource.dosageInstruction[0].timing.event[0]
                    if resource.dosageInstruction[0].timing
                    else np.nan
                )
            except:
                event_time = np.nan
            resource_id = resource.id

            elements = {
                "resource_id": resource_id,
                "medication_id": medication_id,
                "patient_id": patient_id,
                "event_time": event_time,
                "source": source,
                "status": status,
            }

            records.append(elements)

        return records

    @staticmethod
    def extract_medication_administration(bundle):
        records = []
        for entry in bundle.entry or []:
            resource = entry.resource
            try:
                importer = next(
                    (
                        e.valueString
                        for e in resource.meta.extension
                        if e.url == "http://uk-essen.de/fhir/extension-importer-name"
                    ),
                    None,
                )
                if "cato" in importer:
                    source = "cato"
                elif "medico" in importer:
                    source = "medico"
                else:
                    source = np.nan
            except Exception as e:
                source = np.nan

            status = resource.status

            medication_id = (
                resource.medicationReference.reference.split("Medication/")[-1]
                if resource.medicationReference is not None
                else None
            )
            patient_id = resource.subject.reference.split("Patient/")[-1]
            try:
                event_time = resource.effectiveDateTime
            except Exception as e:
                event_time = resource.effectivePeriod.start
            except Exception as e:
                event_time = pd.NaT

            resource_id = resource.id

            elements = {
                "resource_id": resource_id,
                "medication_id": medication_id,
                "patient_id": patient_id,
                "event_time": event_time,
                "source": source,
                "status": status,
            }
            records.append(elements)

        return records

    # Build Medication DataFrame
    def build_medication(self):
        base_dict = transform_dict_constants()

        # only pull medications if they are older than 30 days in live prediciton
        if (
            self.config["is_live_prediction"]
            and Path(self.config["medication_path_raw"]).exists()
        ):
            modified_date = datetime.datetime.fromtimestamp(
                os.path.getmtime(self.config["medication_path_raw"])
            )
            duration = datetime.datetime.today() - modified_date
            re_download = True if duration.days > 30 else False
        else:
            re_download = True

        # 1. Get all medications by ID
        if re_download:
            df_medications = self.search.sail_through_search_space_to_dataframe(
                process_function=self.extract_medication,
                request_params=self.config["medication_params"],
                resource_type="Medication",
                time_attribute_name="_lastUpdated",
                date_init="01-01-2000",
                date_end=datetime.datetime.now().date(),
            )
            df_medications.columns = df_medications.columns.astype(str)

            # Simplify medication labels e.g. Temozolomid HEXAL 100mg -> Temozolomid
            df_medications["substance"] = np.nan
            df_medications.dropna(subset=["medicationName"], inplace=True)
            # translating brand names to acutal substances -> keys in medication dict
            df_medications["substance"] = df_medications.medicationName.apply(
                lambda x: [
                    [
                        med
                        for med in sum(base_dict["MEDICATION_DICT_LIST"].values(), [])
                        if med.lower() in x.lower()
                    ]
                    or [np.nan]
                ][0]
            )

            df_medications.dropna(subset=["substance"]).reset_index(
                inplace=False, drop=True
            ).to_feather(self.config["medication_path_raw"])
        else:
            df_medications = pd.read_feather(self.config["medication_path_raw"])

        # 2. MedicationRequest by medications in scope
        if self.config["is_live_prediction"]:
            # todo verify in live before prod live prediction
            dates = {"date": "ge" + str(self.config["start_datetime"])}
        else:
            dates = {}
        self.config["medication_req_params"] = (
            self.config["medication_req_params"] | dates
        )

        df_request = self.search.sail_through_search_space_to_dataframe(
            process_function=self.extract_medication_request,
            resource_type="MedicationRequest",
            request_params=self.config["medication_req_params"],
            time_attribute_name="date",
            date_init=self.config["start_datetime"],
            date_end=datetime.datetime.now().date(),
        )

        df_req_filtered = df_medications.merge(
            df_request, on="medication_id", how="inner"
        )

        # 3. MedicationAdministration by medications in scope
        if self.config["is_live_prediction"]:
            # todo verify in live before prod live prediction
            dates = {"_lastUpdated": "ge" + str(self.config["start_datetime"])}
        else:
            dates = {}
        self.config["medication_admin_params"] = (
            self.config["medication_admin_params"] | dates
        )
        df_admin = self.search.trade_rows_for_dataframe(
            df=df_medications,
            process_function=self.extract_medication_administration,
            request_params=self.config["medication_admin_params"],
            df_constraints={"medication": "medication_id"},
            resource_type="MedicationAdministration",
        )
        df_admin_filtered = df_medications.merge(
            df_admin, on="medication_id", how="inner"
        )

        df_merged = pd.concat([df_req_filtered, df_admin_filtered], axis=0)

        df_merged.lastUpdated = df_merged.lastUpdated.astype(str)
        df_merged.event_time = df_merged.event_time.astype(str)
        df_merged.reset_index(drop=True, inplace=False).to_feather(
            self.config["medication_merged_path"]
        )

    def build_filter_patient_parents(self):
        encs = pd.read_feather(self.config["encounter_path_filtered"])
        encs = encs.drop_duplicates(subset=["patient_id"], keep="first")

        secondary_df = (
            self.search.trade_rows_for_dataframe(
                df=encs,
                df_constraints={"link": "patient_id"},
                resource_type="Patient",
                fhir_paths=[
                    ("linked_patient_id", "link.other.reference"),
                    ("meta_patient", "id"),
                ],
                with_ref=True,
            )
            .explode("linked_patient_id")
            .replace({"Patient/": ""}, regex=True)
        ).drop_duplicates()

        secondary_df.reset_index(drop=True).to_feather(
            self.config["patient_parent_path"]
        )

        merged_df = pd.merge(
            encs[["patient_id"]], secondary_df, on="patient_id", how="outer"
        )

        grouped_patients = self.search.smash_rows(
            merged_df,
            group_by_col="patient_id",
            separator=",",
            unique=True,
            sort=True,
        )

        grouped_patients["linked_patient_id"] = grouped_patients.apply(
            lambda x: x.linked_patient_id
            if not pd.isnull(x.linked_patient_id)
            else x.patient_id,
            axis=1,
        )

        grouped_patients.reset_index(drop=True).to_feather(
            self.config["patient_parent_path_filtered"]
        )

    def build_observation(self):
        query = text(
            """SELECT observation_code_coding.display, value, unit,
        lower("effectiveDateTime")::timestamp as effectiveDateTime,
        replace(observation_subject.reference, 'Patient/', '') as patient_id,
        replace(observation_encounter.reference, 'Encounter/', '') as encounter_id
        FROM observation
        JOIN observation_code_coding ON observation._id = observation_code_coding._resource
        JOIN observation_subject ON observation._id = observation_subject._resource
        JOIN observation_encounter ON observation._id = observation_encounter._resource
        JOIN "observation_valueQuantity" ON observation._id = "observation_valueQuantity"._resource
        """
        )

        with engine.connect() as connection:
            df = pd.read_sql_query(query, connection)
            df.to_feather(self.config["observation_path"])
        exit()


# Class FHIRExtractor
class FHIRFilter:
    """
    desc

    Method:
        -
    Run:

    """

    def __init__(self, config):
        self.config = config

    @staticmethod
    def filter_date(
        start: datetime, end: datetime, resource: pd.DataFrame, date_col: str
    ) -> pd.DataFrame:
        df = resource[
            ((start <= resource[date_col]) & (resource[date_col] <= end))
        ].sort_values([date_col])

        return df

    def filter_encounter(self) -> None:
        enc = pd.read_feather(self.config["encounter_path"])
        enc_filtered = enc[
            [
                "id",
                "meta_extension_0_valueString",
                "status",
                "class_code",
                "type_0_coding_0_code",
                "type_2_coding_0_display",
                "serviceType_coding_0_display",
                "subject_reference",
                "participant_0_type_0_coding_0_display",
                "participant_0_individual_reference",
                "period_start",
                "period_end",
                "hospitalization_admitSource_coding_0_display",
            ]
        ]
        enc_filtered.dropna(inplace=True)
        enc_filtered.columns = [
            "encounter_id",
            "type",
            "status",
            "v3-ActCode",
            "kontaktebene",
            "kontaktart-de",
            "fachabteilungsschluessel",
            "patient_id",
            "v3-ParticipantTypeCode",
            "practicioner_id",
            "start",
            "end",
            "aufnahmeanlass",
        ]

        enc_filtered = enc_filtered[enc_filtered["type"].str.contains("Case|Stay")]
        enc_filtered["type"] = [x.split(".")[-1] for x in enc_filtered["type"]]
        enc_filtered["patient_id"] = enc_filtered["patient_id"].str.split("/").str[-1]
        enc_filtered["practicioner_id"] = (
            enc_filtered["practicioner_id"].str.split("/").str[-1]
        )

        enc_filtered["start"] = pd.to_datetime(enc_filtered["start"])
        enc_filtered["end"] = pd.to_datetime(enc_filtered["end"])

        # Filter encounters with duration <= 2 days
        enc_filtered = enc_filtered[
            enc_filtered["end"] - enc_filtered["start"] > pd.Timedelta(days=2)
        ]

        # Filter to keep only case encounters
        enc_filtered = enc_filtered[enc_filtered["type"] == "Case"]
        # Filter to keep only inpatient encounters (stationary clinic encounters)
        enc_filtered = enc_filtered[enc_filtered["v3-ActCode"] == "IMP"]
        # Take only 50 % of the encounters
        if self.config["task"]:
            valid_pats = enc_filtered[
                enc_filtered["patient_id"].str.startswith(
                    tuple(map(str, [8, 9, "a", "b", "c", "d", "e", "f"]))
                )
            ]["patient_id"]
        else:
            valid_pats = enc_filtered[
                enc_filtered["patient_id"].str.startswith(
                    tuple(map(str, [0, 1, 2, 3, 4, 5, 6, 7]))
                )
            ]["patient_id"]
        enc_filtered = enc_filtered[enc_filtered["patient_id"].isin(valid_pats)]

        enc_filtered.reset_index(drop=True).to_feather(
            self.config["encounter_path_filtered"]
        )

    def filter_patient(self):
        pat = pd.read_feather(self.config["patient_path"])

        # Insurance
        query_all = text(
            """select xp.id, xc2.code as "insurance_type_2"
            from patient xp
            join patient_identifier_type_coding xi on xp._id = xi._resource
            join fhirql_codes xc on xi.system = xc.id and xc.code = 'http://fhir.de/CodeSystem/identifier-type-de-basis'
            join fhirql_codes xc2 on xi.code = xc2.id;
        """
        )

        with engine.connect() as connection:
            df_insurance = pd.read_sql_query(query_all, connection)

        pat = pat.merge(df_insurance, left_on="patient_id", right_on="id", how="left")
        pat.drop(columns=["insurance_type"], inplace=True)
        pat.rename(columns={"insurance_type_2": "insurance_type"}, inplace=True)

        # Get final df
        pat_filtered = pat[["patient_id", "gender", "insurance_type", "birthDate"]]
        pat_filtered["gender"] = [
            "female" if x == 231 else "male" for x in pat_filtered["gender"]
        ]
        mapping_dict = {"PKV": "privat", "GKV": "gesetzlich"}
        pat_filtered["insurance_type"] = (
            pat_filtered["insurance_type"].map(mapping_dict).fillna("unbekannt")
        )

        pat_filtered.reset_index(drop=True).to_feather(
            self.config["patient_path_filtered"]
        )

    @staticmethod
    def to_datetime(df, col_format):
        for k, v in tqdm(col_format.items(), desc=f"Converting DateTime"):
            df[k] = pd.to_datetime(df[k], format=v, utc=True, errors="coerce")
        return df

    @staticmethod
    def filter_by_meta_patients(df) -> pd.DataFrame:
        # filtering out patients that are not in the patient table
        pats = pd.read_feather(
            "/local/work/merengelke/ship_former/data_filtered/patient_parents.ftr"
        )
        pats_list = pats["linked_patient_id"].tolist()

        def split_list_elements(input_list):
            output_list = []
            for item in input_list:
                if "," in item:
                    output_list.extend(item.split(","))
                else:
                    output_list.append(item)
            return output_list

        pats_list2 = split_list_elements(pats_list)

        return df[df["patient_id"].isin(pats_list2)]

    def filter_procedures(self) -> None:
        pros = pd.read_feather(self.config["procedure_path"])

        pros = self.filter_by_meta_patients(pros)

        pros_filtered = pros.dropna(subset=["encounter_id"])
        pros_filtered = pros_filtered[
            (pros_filtered["status"] == "completed")
            | (pros_filtered["status"] == "in-progress")
            | (pros_filtered["status"] == "preparation")
        ]

        if not self.config["is_live_prediction"]:
            pros_filtered["procedure_start"] = col_to_datetime(
                pros_filtered["procedure_start"]
            )
            pros_filtered["procedure_end"] = col_to_datetime(
                pros_filtered["procedure_end"]
            )
            pros_filtered = self.filter_date(
                self.config["start_datetime"],
                self.config["end_datetime"],
                pros_filtered,
                "procedure_start",
            )

        pros_filtered.reset_index(drop=True).to_feather(
            self.config["procedure_path_filtered"]
        )

    def filter_conditions(self) -> None:
        conds = pd.read_feather(self.config["condition_path"])
        conds = conds[
            [
                "subject_reference",
                "encounter_reference",
                "id",
                "recorder_reference",
                "recordedDate",
                "code_coding_0_code",
                "code_coding_0_display",
                "code_coding_0_version",
                "category_0_coding_0_code",
                "category_1_coding_0_code",
                "category_2_coding_0_code",
                "category_3_coding_0_code",
            ]
        ]
        conds.columns = [
            "patient_id",
            "encounter_id",
            "condition_id",
            "practitioner_recoder_id",
            "condition_date",
            "icd_code",
            "icd_display",
            "icd_version",
            "code_entlassungsdiagnose",
            "code_drg_hauptdiagnose",
            "code_adm_hauptdiagnose",
            "code_med_hauptdiagnose",
        ]
        conds = conds.dropna(subset=["encounter_id", "patient_id"])

        conds["patient_id"] = conds["patient_id"].str.split("/").str[-1]
        conds = self.filter_by_meta_patients(conds)

        conds["encounter_id"] = conds["encounter_id"].str.split("/").str[-1]
        conds["practitioner_recoder_id"] = (
            conds["practitioner_recoder_id"].str.split("/").str[-1]
        )
        conds["icd_root_code"] = conds["icd_code"].str.split(".").str[0]
        conds["code_med_hauptdiagnose"] = [
            True if x else False for x in conds["code_med_hauptdiagnose"]
        ]

        conds.reset_index(drop=True).to_feather(self.config["condition_path_filtered"])


# Class FHIRExtractor
class FHIRValidator:
    """
    desc

    Method:
        -
    Run:

    """

    def __init__(self, config):
        self.config = config

    @staticmethod
    def na_checker(field_name: str, na_counts: pd.Series, is_error: bool) -> None:
        if na_counts[field_name] and is_error:
            logging.error(f"At least one {field_name} is zero")
            raise ValueError(f"At least one {field_name} is zero")
        elif na_counts[field_name]:
            logging.warning(f"At least one {field_name} is zero")
        else:
            logging.info(f"Validation for {field_name} passed")

    def validate_encounters(self) -> None:
        # todo check if any encounters codes do now follow the encoding rules
        return None

    def validate_patient(self) -> None:
        pats = pd.read_feather(self.config["patient_path_filtered"])
        na_counts = pats.isna().sum()
        self.na_checker("patient_id", na_counts, True)
        self.na_checker("gender", na_counts, True)

    def validate_bdp(self) -> None:
        bdp = pd.read_feather(self.config["bdp_path_filtered"])
        bdp["ausgabe_datetime"] = col_to_datetime(bdp.date)
        bdp_count = bdp.ausgabe_datetime.value_counts().sort_index()[:-1]
        if (bdp_count == 0).any():
            logging.warning("BDP count for one or more imported days = 0")

    def validate_observations(self) -> None:
        obs = pd.read_feather(self.config["obs_path_filtered"])
        na_counts = obs.isna().sum()
        self.na_checker("patient_id", na_counts, True)
        self.na_checker("value", na_counts, True)

    def validate_procedures(self) -> None:
        pros = pd.read_feather(self.config["procedure_path_filtered"])
        if pros.patient_id.isna().sum() != 0:
            logging.warning("Some patient ids are na")

    def validate_conditions(self) -> None:
        conds = pd.read_feather(self.config["condition_path_filtered"])
        if conds.patient_id.isna().sum() != 0:
            logging.warning("Some patient ids are na")

    def validate_medicaitons(self) -> None:
        meds = pd.read_feather(self.config["medication_merged_path_filtered"])
        na_counts = meds.isna().sum()
        self.na_checker("medicationName", na_counts, True)
        self.na_checker("patient_id", na_counts, True)
        self.na_checker("event_time", na_counts, True)
