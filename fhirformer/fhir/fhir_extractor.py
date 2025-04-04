import base64
import logging
import multiprocessing
import os
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import pandas as pd
from fhir_pyrate import Ahoy
from fhir_pyrate.pirate import Pirate
from sqlalchemy import text
from tqdm import tqdm

from fhirformer.fhir.functions import extract_bdp, extract_imaging_study
from fhirformer.fhir.util import (
    FOLDER_DEPTH,
    OUTPUT_FORMAT,
    auth,
    check_and_read,
    engine,
    extract_id,
    get_category_name,
    get_document_path,
    get_text,
    group_meta_patients,
    load_df,
    store_df,
)
from fhirformer.helper.util import timed

logger = logging.getLogger(__name__)


class FHIRExtractor:
    def __init__(self, config):
        self.config = config
        # Pirate
        self.search = Pirate(
            auth=auth,
            base_url=os.environ["SEARCH_URL"],
            num_processes=90,
            print_request_url=False,
            cache_folder=Path(self.config["root_dir"] / "cache")
            if self.config["step"] != "inference"
            else None,
        )

    def skip_build(self, path: Path):
        if self.config["rerun_cache"] or not path.exists():
            return False
        else:
            if path.exists():
                logger.info(
                    f"Skipping build, file {str(path).split('/')[-1]} already exists."
                )
            return path.exists()

    @timed
    def build(self, resource: str):
        if resource == "condition":
            self.build_condition()
        elif resource == "encounter":
            self.build_encounter_raw()
            self.build_encounter()
        elif resource == "biologically_derived_product":
            self.build_bdp()
        elif resource == "patient":
            self.build_initial_patient()
            self.build_patient()
        elif resource == "procedure":
            self.build_procedure()
        elif resource == "observation":
            self.build_observation()
        elif resource == "imaging_study":
            self.build_imaging_study()
        elif resource == "diagnostic_report":
            self.build_diagnostic_report()
            if self.config["download_documents"]:
                self.download_documents()
        elif resource == "episode_of_care":
            self.build_episode_of_care()
        elif resource == "service_request":
            self.build_service_request()
        elif resource == "medication":
            self.build_medication()
        else:
            raise NotImplementedError(f"Resource {resource} not supported")

    @timed
    def df_from_query(self, query: str, chunk_size: int = 1000) -> pd.DataFrame:
        query_all = query.strip()
        logger.debug(f"Running query: {query_all}")
        dfs = []
        with engine.connect() as connection:
            # Chunk connection
            start = time.perf_counter()
            for chunk in tqdm(
                pd.read_sql_query(
                    sql=text(query_all),
                    con=connection.execution_options(stream_results=True),
                    chunksize=chunk_size,
                )
            ):
                logger.debug(f"Query relative {time.perf_counter() - start} seconds")
                dfs.append(chunk)
        return pd.concat(dfs, ignore_index=True)

    def check_and_build_file(self, name: str) -> pd.DataFrame:
        file_path = self.config["data_dir"] / f"{name}{OUTPUT_FORMAT}"
        if not file_path.exists():
            if name == "encounter_raw":
                logger.info("The encounter_raw file, building it")
                self.build_encounter_raw()
            elif name == "encounter":
                logger.info("The encounter file was not found, building it")
                self.build_encounter()
            elif name == "patient":
                logger.info("The patient file was not found, building it")
                self.build_patient()
            else:
                raise ValueError(f"Name {name} not recognized for building")
        df = check_and_read(file_path)
        return df

    def build_initial_patient(self):
        output_path = self.config["data_dir"] / f"initial_patient{OUTPUT_FORMAT}"
        if self.skip_build(output_path):
            return

        # Encounters define the base cohort
        base_df = load_df(
            self.config["data_dir"] / f"encounter{OUTPUT_FORMAT}", "Encounter"
        )

        logging.info(f"numb of patients: {len(base_df)}")
        df = self.search.trade_rows_for_dataframe(
            df=base_df,
            resource_type="Patient",
            df_constraints=self.config["patient_constraints"],
            request_params=self.config["patient_params"],
        )

        if isinstance(df, dict):
            df = pd.concat(df.values()) if df else pd.DataFrame()

        df.drop_duplicates(subset=["patient_id"], inplace=True)
        logging.info(f"numb of patients: {len(df)}")
        self.store_pyrate_extraction(df, "initial_patient")

    def build_patient(self):
        output_path = self.config["data_dir"] / f"patient{OUTPUT_FORMAT}"
        if self.skip_build(output_path):
            return

        pats = self.check_and_build_file("initial_patient")

        link_column = pats.columns[pats.columns.str.contains("link")]
        link_reference_column = link_column[link_column.str.contains("reference")]
        other_list = pats[link_reference_column].apply(
            lambda x: ",".join(
                [item.replace("Patient/", "") for item in x.dropna().astype(str)]
            ),
            axis=1,
        )

        pats = pats[
            [
                "id",
                "birthDate",
                "gender",
                "identifier_1_type_coding_0_code",
                "deceasedDateTime",
            ]
        ]

        pats.columns = [
            "patient_id",
            "birth_date",
            "sex",
            "insurance_type",
            "deceased_date",
        ]

        pats["other_list"] = other_list
        pats = pats.explode("other_list")
        metas = pats[~pats["other_list"].isna()]
        union = pd.merge(
            how="left",
            left=pats,
            right=metas,
            left_on="patient_id",
            right_on="other_list",
            suffixes=("", "_meta"),
        )
        union.drop(
            columns=["other_list_meta"],
            inplace=True,
        )
        assert len(union[union["patient_id"].isna()]) == 0

        non_metas = union[union["patient_id_meta"].isna()].copy()
        non_metas["linked_patient_id"] = non_metas["patient_id"]
        non_metas.drop(
            columns=[
                "other_list",
                "patient_id_meta",
                "birth_date_meta",
                "sex_meta",
                "insurance_type_meta",
                "deceased_date_meta",
            ],
            inplace=True,
        )
        union.dropna(subset="patient_id_meta", inplace=True)
        # So here all the patients that have a meta in common are put together in one single row
        with multiprocessing.Pool(30) as pool:
            results = pool.map(
                group_meta_patients,
                union.groupby(by="patient_id_meta"),
            )
        # The results are transformed to a dataframe
        df = pd.DataFrame([ss for s in results for ss in s])
        # We join them with the ones from before that we are sure that do not have any metas
        df = pd.concat([df, non_metas], ignore_index=True)
        df.drop_duplicates(["patient_id"], keep="first", inplace=True)
        store_df(df, output_path, "Patient")

    def store_pyrate_extraction(
        self,
        df: pd.DataFrame,
        output_name: str,
        timestampt_columns: List[str] = None,
        store: bool = True,
    ):
        output_path = self.config["data_dir"] / f"{output_name}{OUTPUT_FORMAT}"
        resource_name = output_name.title().replace("_", "")
        if self.skip_build(output_path):
            return
        for col in timestampt_columns or []:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        logger.info(f"Extracted {len(df)} {resource_name}s")
        if store:
            store_df(df, output_path, resource_name)
        else:
            return df

    def default_pyrate_extraction(
        self,
        output_name: str,
        process_function: Callable = None,
        fhir_paths: Union[
            List[str], List[Tuple[str, str]], List[Union[str, Tuple[str, str]]]
        ] = None,
        request_params: Dict[str, str] = None,
        time_attribute_name: str = None,
        explode: List = None,
        disable_parallel: bool = False,
    ):
        output_path = self.config["data_dir"] / f"{output_name}{OUTPUT_FORMAT}"
        resource_name = output_name.title().replace("_", "")
        if self.skip_build(output_path):
            return
        params = {
            "resource_type": resource_name,
        }
        if fhir_paths is not None:
            params["fhir_paths"] = fhir_paths
        elif process_function is not None:
            params["process_function"] = process_function
        if disable_parallel:
            new_request_params = request_params.copy()
            new_request_params[time_attribute_name] = (
                f"ge{self.config['start_datetime']}",
                f"le{self.config['end_datetime']}",
            )
            params["request_params"] = new_request_params
            df = self.search.steal_bundles_to_dataframe(**params)
        else:
            params.update(
                dict(
                    request_params=request_params,
                    time_attribute_name=time_attribute_name,
                    date_init=self.config["start_datetime"],
                    date_end=self.config["end_datetime"],
                )
            )
            df = self.search.sail_through_search_space_to_dataframe(
                **params,
            )

        if isinstance(df, dict):
            df = pd.concat(df.values()) if df else pd.DataFrame()

        if explode:
            df = df.explode(explode)

        store_df(df, output_path, resource_name)

    def build_encounter_raw(self):
        output_name = "encounter_raw"
        output_path = self.config["data_dir"] / f"{output_name}{OUTPUT_FORMAT}"
        if self.skip_build(output_path):
            return

        if self.config["step"] == "live_inference":
            self.config["encounter_params"]["status"] = "in-progress"

        df = self.search.sail_through_search_space_to_dataframe(
            resource_type="Encounter",
            request_params=self.config["encounter_params"],
            time_attribute_name="date",
            date_init=self.config["start_datetime"],
            # For live inference, ongoing encounters with unknown end date are dated to 2099-01-01
            date_end=self.config["end_datetime"]
            if not self.config["live_inference"]
            else "2099-01-01",
            fhir_paths=[
                ("encounter_id", "id"),
                ("patient_id", "subject.reference.replace('Patient/', '')"),
                ("status", "status"),
                ("v3_act_code", "class.code"),
                ("v3_act_code_display", "class.display"),
                ("type_code", "type[0].coding[0].code"),
                ("type_display", "type[0].coding[0].display"),
                ("fachabteilungsschluessel_code", "serviceType.coding[0].code"),
                ("fachabteilungsschluessel", "serviceType.coding[0].display"),
                ("v3_ParticipantTypeCode_code", "participant[0].type.coding[0].code"),
                (
                    "v3_ParticipantTypeCode_display",
                    "participant[0].type.coding[0].display",
                ),
                ("practitioner_id", "participant[0].individual.reference"),
                ("start", "period.start"),
                ("end", "period.end"),
                ("aufnahmeanlass_code", "hospitalization.admitSource.coding[0].code"),
                (
                    "aufnahmeanlass_display",
                    "hospitalization.admitSource.coding[0].display",
                ),
                (
                    "discharge_text",
                    "hospitalization.dischargeDisposition.extension[0].extension[0].valueCoding.code",
                ),
                (
                    "discharge_place_code",
                    "hospitalization.dischargeDisposition.extension[0].extension[0].valueCoding.display",
                ),
                (
                    "discharge_place_display",
                    "hospitalization.dischargeDisposition.coding[0].display",
                ),
                ("kind", "meta.extension[1].valueString"),
            ],
        )

        if isinstance(df, dict):
            df = pd.concat(df.values()) if df else pd.DataFrame()

        store_df(df, output_path, "encounter_raw")

    def build_encounter(self, minimum_days: int = 2) -> None:
        output_path = self.config["data_dir"] / f"encounter{OUTPUT_FORMAT}"
        if self.skip_build(output_path):
            return
        enc_filtered = self.check_and_build_file("encounter_raw")

        enc_filtered.drop_duplicates(
            subset=["encounter_id", "type_code", "v3_act_code"],
            inplace=True,
        )

        # Convert to datetime and ensure timezone awareness more robustly
        enc_filtered["start"] = pd.to_datetime(enc_filtered["start"], utc=True)
        enc_filtered["end"] = pd.to_datetime(enc_filtered["end"], utc=True)

        # Filter encounters with duration <= 2 days
        enc_filtered = enc_filtered[
            enc_filtered["end"] - enc_filtered["start"]
            > pd.Timedelta(days=minimum_days)
        ]

        enc_filtered["patient_id"] = extract_id(enc_filtered.patient_id)
        enc_filtered["practitioner_id"] = extract_id(enc_filtered.practitioner_id)

        store_df(enc_filtered, output_path, "Encounter")

    def build_bdp(self):
        output_path = (
            self.config["data_dir"] / f"biologically_derived_product{OUTPUT_FORMAT}"
        )
        if self.skip_build(output_path):
            return

        df = self.search.sail_through_search_space_to_dataframe(
            process_function=extract_bdp,
            resource_type="BiologicallyDerivedProduct",
            request_params=self.config["bdp_params"],
            time_attribute_name="shipStorageEnd",
            date_init=self.config["start_datetime"],
            date_end=self.config["end_datetime"],
        )

        if isinstance(df, dict):
            # If df is a dictionary, extract the DataFrame we need
            df = pd.concat(df.values()) if df else pd.DataFrame()

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
                "display_y",
                "output_to_y",
            ],
            axis=1,
            inplace=True,
        )
        merged.columns = [
            "bdp_id",
            "service_request_id",
            "ausgabe_datetime",
            "ausgabe_type",
            "code",
            "display",
            "output_to",
            "patient_id",
            "output_to_einskurz",
            "output_to_einscode",
        ]

        store_df(
            merged,
            self.config["data_dir"] / f"biologically_derived_product{OUTPUT_FORMAT}",
            "BiologicallyDerivedProduct",
        )

    def build_procedure(self):
        self.default_pyrate_extraction(
            "procedure",
            request_params=self.config["procedure_params"],
            time_attribute_name="date",
            fhir_paths=[
                ("procedure_id", "id"),
                ("patient_id", "subject.reference.replace('Patient/', '')"),
                ("encounter_id", "encounter.reference.replace('Encounter/', '')"),
                ("code", "code.coding.code"),
                ("status", "status"),
                ("version", "code.coding.version"),
                ("display", "code.coding.display"),
                ("effectivedatetimestart_v1", "performedPeriod.start"),
                ("effectivedatetimeend_v1", "performedPeriod.end"),
                ("effectivedatetimestart_v2", "performedDateTime"),
                ("effectivedatetimeend_v2", "performedDateTime"),
                ("location_id", "location.reference.replace('Location/', '')"),
            ],
        )

    # Building Condition DataFrame
    def build_condition(self):
        self.default_pyrate_extraction(
            "condition",
            request_params=self.config["condition_params"],
            time_attribute_name="recorded-date",
            fhir_paths=[
                ("condition_id", "id"),
                ("patient_id", "subject.reference.replace('Patient/', '')"),
                ("encounter_id", "encounter.reference.replace('Encounter/', '')"),
                ("practitioner_id", "asserter.reference"),
                ("condition_date", "recordedDate"),
                ("icd_code", "code.coding.code"),
                ("icd_display", "code.coding.display"),
                ("icd_version", "code.coding.version"),
                ("code_diagnosis_type", "category.coding.code"),
                ("code_diagnosis_display", "category.coding.display"),
            ],
        )

    def build_meta_patient_pyrate(self):
        output_path = self.config["data_dir"] / f"meta_patient{OUTPUT_FORMAT}"
        if self.skip_build(output_path):
            return
        encs = self.check_and_build_file("encounter")
        encs = encs.drop_duplicates(subset=["patient_id"], keep="first")

        secondary_df = self.search.trade_rows_for_dataframe(
            df=encs,
            df_constraints={"link": "patient_id"},
            resource_type="Patient",
            fhir_paths=[
                ("linked_patient_id", "link.other.reference"),
                ("meta_patient", "id"),
            ],
            with_ref=True,
        )

        if isinstance(secondary_df, dict):
            secondary_df = (
                pd.concat(secondary_df.values()) if secondary_df else pd.DataFrame()
            )

        secondary_df = (
            secondary_df.explode("linked_patient_id").replace(
                {"Patient/": ""}, regex=True
            )
        ).drop_duplicates()

        secondary_df.reset_index(drop=True).to_feather(
            self.config["data_dir"] / f"meta_patient{OUTPUT_FORMAT}"
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
            lambda x: (
                x.linked_patient_id
                if not pd.isnull(x.linked_patient_id)
                else x.patient_id
            ),
            axis=1,
        )

        store_df(grouped_patients, output_path, "MetaPatient")

    def build_observation(self):
        self.default_pyrate_extraction(
            output_name="observation",
            time_attribute_name="date",
            request_params=self.config["obs_params"],
            fhir_paths=[
                ("observation_id", "id"),
                ("effectivedatetime", "effectiveDateTime"),
                "issued",
                ("patient_id", "subject.reference.replace('Patient/', '')"),
                ("encounter_id", "encounter.reference.replace('Encounter/', '')"),
                ("code", "code.coding.code"),
                ("display", "code.coding.display"),
                ("value_quantity", "valueQuantity.value"),
                ("value_unit", "valueQuantity.unit"),
            ],
            disable_parallel=True,
        )

    def build_imaging_study(self):
        self.default_pyrate_extraction(
            output_name="imaging_study",
            time_attribute_name="started",
            request_params={"_sort": "-started"},
            process_function=extract_imaging_study,
            explode=[
                "series_instance_uid",
                "series_number",
                "series_instances",
                "series_description",
                "series_modality",
                "series_modality_version",
                "series_body_site",
            ],
        )

    def build_diagnostic_report(self):
        self.default_pyrate_extraction(
            "diagnostic_report",
            request_params=self.config["diagnostic_report_params"],
            time_attribute_name="issued",
            fhir_paths=[
                ("id", "diagnotic_report_id"),
                ("effective_datetime", "effectiveDateTime"),
                "issued",
                ("patient_id", "subject.reference.replace('Patient/', '')"),
                ("encounter_id", "encounter.reference.replace('Encounter/', '')"),
                (
                    "imaging_study_id",
                    "imagingStudy.reference.replace('ImagingStudy/', '')",
                ),
                ("category", "category.coding.code"),
                ("category_display", "category.coding.display"),
                ("title", "presentedForm.title"),
                ("data", "presentedForm.data"),
                ("content_type", "presentedForm.contentType"),
                ("url", "presentedForm.url"),
            ],
        )

    def download_documents(self):
        df = check_and_read(
            self.config["data_dir"] / f"diagnostic_report{OUTPUT_FORMAT}"
        )
        logger.info(f"Starting to download {len(df)} documents.")
        document_folder = self.config["data_dir"] / "documents"
        ume_auth = Ahoy(
            auth_method="env",
            username=os.environ["FHIR_USER"],
            auth_url=os.environ["BASIC_AUTH_UME"],
            refresh_url=os.environ["REFRESH_AUTH"],
        )
        failed = []
        for row_dict in tqdm(df.to_dict(orient="records"), total=len(df)):
            txt = None
            category = get_category_name(row_dict["category_display"])
            document_path = get_document_path(
                root_path=document_folder / category,
                filename=f"{row_dict['diagnostic_report_id']}.txt",
                folder_depth=FOLDER_DEPTH,
            )
            if document_path.exists():
                continue
            # Some documents, like radiology documents,
            # have a data field where the document is stored
            if row_dict["data"] is not None and len(row_dict["data"]) > 0:
                assert len(row_dict["data"]) == 1, row_dict["diagnostic_report_id"]
                txt = base64.b64decode(row_dict["data"][0]).decode()
            # Other documents have an url with different formats
            elif row_dict["url"] is not None and len(row_dict["url"]) > 0:
                # Iterate over the formats that we can handle, in order of preference
                for choice in ["text", "word", "pdf"]:
                    # Choose the format, if it is available in the content types
                    chosen_id = next(
                        iter(
                            i
                            for i in range(len(row_dict["content_type"])) or []
                            if choice in row_dict["content_type"][i]
                        ),
                        None,
                    )
                    # If the format exists, try to download it
                    if chosen_id is not None:
                        txt = get_text(
                            ume_auth.session,
                            row_dict["url"][chosen_id],
                            content=row_dict["content_type"][chosen_id],
                            row_for_debug=row_dict,
                        )
                        # If we managed to download something, break out of the loop
                        if txt is not None:
                            break
            else:
                failed.append(row_dict)
            if txt is None:
                failed.append(row_dict)
                continue
            document_path.parent.mkdir(parents=True, exist_ok=True)
            with document_path.open("w") as fp:
                fp.write(txt)
        pd.DataFrame(failed).to_pickle(document_folder / "failed.pkl")

    def build_episode_of_care(self):
        self.default_pyrate_extraction(
            "episode_of_care",
            request_params=self.config["episode_of_care_params"],
            time_attribute_name="date",
            fhir_paths=[
                "id",
                ("start", "period.start"),
                ("end", "period.end"),
                ("patient_id", "patient.reference.replace('Patient/', '')"),
                (
                    "cert",
                    "extension.where(url = 'https://uk-essen.de/HIS/Cerner/Medico/TumorDocumentation/Onkozert').valueString",
                ),
                (
                    "primary_case",
                    "extension.where(url = 'https://uk-essen.de/HIS/Cerner/Medico/TumorDocumentation/Primärfall').valueBoolean",
                ),
                (
                    "primary_year",
                    "extension.where(url = 'https://uk-essen.de/HIS/Cerner/Medico/TumorDocumentation/Primärfalljahr').valueString",
                ),
                (
                    "first_diagnosis_date",
                    "extension.where(url = 'http://uk-koeln.de/fhir/StructureDefinition/Extension/nNGM/Erstdiagnose').valueDateTime",
                ),
                (
                    "treatment_program",
                    "extension.where(url = 'https://uk-essen.de/HIS/Cerner/Medico/TumorDocumentation/TumorIdentifier').extension.where(url = 'https://uk-essen.de/HIS/Cerner/Medico/TumorDocumentation/TumorIdentifier/Bogen').valueString",
                ),
            ],
        )

    def build_service_request(self):
        output_path = self.config["data_dir"] / f"service_request{OUTPUT_FORMAT}"
        if self.skip_build(output_path):
            return

        self.default_pyrate_extraction(
            "service_request",
            request_params=self.config["service_request_params"],
            time_attribute_name="authored",
            fhir_paths=[
                "id",
                "status",
                "intent",
                "priority",
                "authoredOn",
                ("patient_id", "subject.reference.replace('Patient/', '')"),
                (
                    "based_service_request_id",
                    "basedOn.reference.replace",
                ),
                ("code", "code.coding.code"),
                ("code_display", "code.coding.display"),
                ("category_code", "category.coding.code"),
                ("category_display", "category.coding.display"),
            ],
        )

    # Build Medication DataFrame
    def build_medication(self):
        output_path = self.config["data_dir"] / f"medication{OUTPUT_FORMAT}"
        if self.skip_build(output_path):
            return

        # 1. MedicationAdministration by medications in scope
        df_admin = self.search.sail_through_search_space_to_dataframe(
            request_params=self.config["medication_admin_params"],
            resource_type="MedicationAdministration",
            time_attribute_name="effective-time",
            date_init=self.config["start_datetime"],
            date_end=self.config["end_datetime"],
            fhir_paths=[
                (
                    "medication_id",
                    "medicationReference.reference.replace('Medication/', '')",
                ),
                ("patient_id", "subject.reference.replace('Patient/', '')"),
                ("encounter_id", "context.reference.replace('Encounter/', '')"),
                ("status", "status"),
                ("datetime", "effectiveDateTime"),
            ],
        )

        if isinstance(df_admin, dict):
            df_admin = pd.concat(df_admin.values()) if df_admin else pd.DataFrame()

        df_admin.dropna(subset=["medication_id"], inplace=True)

        # 2. Get all medications to get the medication name
        df_medications = self.search.trade_rows_for_dataframe(
            df=df_admin,
            df_constraints={"_id": "medication_id"},
            resource_type="Medication",
            fhir_paths=[
                ("medication_id", "_id"),
                ("medication", "code.text"),
            ],
        )

        if isinstance(df_medications, dict):
            df_medications = (
                pd.concat(df_medications.values()) if df_medications else pd.DataFrame()
            )

        df_admin_filtered = df_medications.merge(
            df_admin, on="medication_id", how="inner"
        )

        self.store_pyrate_extraction(df_admin_filtered, "medication")
