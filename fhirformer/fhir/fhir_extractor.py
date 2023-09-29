import base64
import logging
import multiprocessing
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import pandas as pd
from fhir_pyrate import Ahoy
from fhir_pyrate.pirate import Pirate
from sqlalchemy import text
from tqdm import tqdm

from fhirformer.fhir.functions import extract_bdp, extract_imaging_study
from fhirformer.fhir.util import (
    OUTPUT_FORMAT,
    auth,
    check_and_read,
    engine,
    get_text,
    group_meta_patients,
    reduce_cardinality,
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
            num_processes=30,
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

        # with engine.connect() as connection:
        #     start = time.time()
        #     result = connection.execute(text(query_all))
        #     logger.info(f"Query took {time.time() - start} seconds")
        #     df = pd.DataFrame(result.fetchmany(1000), columns=result.keys())
        #     logger.info(f"Fetched {len(df)} rows in {time.time() - start} seconds")
        # return df

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
        pats = self.df_from_query(
            """
            select
              p.id AS patient_id,
              p._id AS metrics_id,
              lower("birthDate")::timestamp AS birth_date,
              fhirql_read_codes(gender) AS sex,
              fhirql_read_codes(c.code) AS insurance_type,
              ARRAY_AGG(replace(pl.reference, 'Patient/', '')) AS other_list
            FROM patient p
            LEFT JOIN patient_link_other pl ON p._id = pl._resource
            LEFT JOIN patient_identifier_type_coding c ON c._resource = p._id and fhirql_read_codes(c.system) = '{http://fhir.de/CodeSystem/identifier-type-de-basis}'
            GROUP BY p.id, p._id, lower("birthDate")::timestamp, fhirql_read_codes(gender), c.code
            """
        )
        store_df(pats, output_path, "Patient")

    def build_patient(self):
        output_path = self.config["data_dir"] / f"patient{OUTPUT_FORMAT}"
        if self.skip_build(output_path):
            return

        pats = self.check_and_build_file("initial_patient")
        pats["sex"] = reduce_cardinality(pats["sex"], set_to_none=True)
        pats["insurance_type"] = reduce_cardinality(
            pats["insurance_type"], set_to_none=True
        )
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
                "metrics_id",
                "other_list",
                "patient_id_meta",
                "metrics_id_meta",
                "birth_date_meta",
                "sex_meta",
                "insurance_type_meta",
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

    def default_metrics_extraction(
        self,
        output_name: str,
        query: str,
        timestamp_columns: List[str] = None,
        store: bool = True,
    ):
        output_path = self.config["data_dir"] / f"{output_name}{OUTPUT_FORMAT}"
        resource_name = output_name.title().replace("_", "")
        if self.skip_build(output_path):
            return
        df = self.df_from_query(query)
        for col in timestamp_columns or []:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        logger.info(f"Extracted {len(df)} {resource_name}s")
        if store:
            store_df(df, output_path, resource_name)
        else:
            return df

    def large_metrics_extraction(
        self, query_template: str, output_name: str, store: bool = True
    ):
        output_path = self.config["data_dir"] / f"{output_name}{OUTPUT_FORMAT}"
        resource_name = output_name.title().replace("_", "")
        if self.skip_build(output_path):
            return

        time_frame_length = timedelta(days=1)
        start_datetime = pd.to_datetime(self.config["start_datetime"])
        end_datetime = pd.to_datetime(self.config["end_datetime"])

        # Initialize an empty list to store DataFrames
        dfs = []
        counter = 0
        # Generate time frames and query for each
        current_start = start_datetime
        while current_start < end_datetime:
            current_end = current_start + time_frame_length
            query = query_template.format(current_start, current_end)

            # Execute the query and append the resulting DataFrame to the list
            df = self.df_from_query(query, chunk_size=10000)
            logger.info(
                f"Extracted {len(df)} {resource_name}s from time frame {current_start} to {current_end}"
            )
            dfs.append(df)
            counter += len(df)
            current_start = current_end

        final_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Extracted {len(final_df)} {resource_name}s")
        if store:
            store_df(final_df, output_path, resource_name)
        else:
            return final_df

    def default_pyrate_extraction(
        self,
        output_name: str,
        process_function: Callable = None,
        fhir_paths: Union[List[str], List[Tuple[str, str]]] = None,
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
        if explode:
            df = df.explode(explode)

        store_df(df, output_path, resource_name)

    def build_encounter_raw(self):
        self.default_metrics_extraction(
            output_name="encounter_raw",
            query=f"""
            select e.id,
            e._id as metrics_id,
            replace(e_sub.reference, 'Patient/', '') as patient_id,
            fhirql_read_codes(e_ind.system) as kind,
            fhirql_read_codes(e.status) as status,
            e_cs.display as v3_act_code,
            e_cs.code as v3_act_code_display,
            e_tcod.code as type_code,
            e_tcod.display as type_display,
            e_stype.code as fachabteilungsschluessel_code,
            e_stype.display as fachabteilungsschluessel,
            e_ptype.code as v3_ParticipantTypeCode_code,
            e_ptype.display as v3_ParticipantTypeCode_display,
            replace(e_pindi.reference, 'Practitioner/', '') as practitioner_id,
            lower(e_per.start)::timestamp as start,
            lower(e_per.end)::timestamp as end,
            e_hosad.code as aufnahmeanlass_code,
            e_hosad.display as aufnahmeanlass_display,
            e_dis.text as discharge_text,
            e_dis_code.code as discharge_place_code,
            e_dis_code.display as discharge_place_display
            from encounter e
            join encounter_subject e_sub on e_sub._resource = e._id
            join encounter_identifier e_ind on e_ind._resource = e._id
            join encounter_class e_cs on e_cs._resource = e._id and e_cs.system = 16
            left join encounter_type_coding e_tcod on e_tcod._resource = e._id
            left join "encounter_serviceType_coding" e_stype on e_stype._resource = e._id
            left join encounter_participant_type_coding e_ptype on e_ptype._resource = e._id
            left join encounter_participant_individual e_pindi on e_pindi._resource = e._id
            join encounter_period e_per on e_per._resource = e._id
            left join "encounter_hospitalization_admitSource_coding" e_hosad on e_hosad._resource = e._id and e_hosad.system = 107
            left join "encounter_hospitalization_dischargeDisposition" e_dis on e_dis._resource = e._id
            left join "encounter_hospitalization_dischargeDisposition_coding" e_dis_code on e_dis_code._resource = e._id
            where e_per.start && tstzrange('{self.config["start_datetime"]}'::date, '{self.config["end_datetime"]}'::date, '[)');
            """,
        )

    def build_encounter(self, minimum_days: int = 2) -> None:
        output_path = self.config["data_dir"] / f"encounter{OUTPUT_FORMAT}"
        if self.skip_build(output_path):
            return
        enc_filtered = self.check_and_build_file("encounter_raw")

        enc_filtered.drop_duplicates(
            subset=["id", "metrics_id", "type_code", "v3_act_code"], inplace=True
        )

        enc_filtered["kind"] = enc_filtered["kind"].apply(lambda x: x[0] if x else None)
        # split code at last / an keep last element
        enc_filtered["kind"] = enc_filtered["kind"].apply(
            lambda x: x.split("/")[-1] if x else None
        )
        enc_filtered["status"] = enc_filtered["status"].apply(
            lambda x: x[0] if x else None
        )

        enc_filtered = enc_filtered.loc[
            ~enc_filtered["kind"].isna()
            & enc_filtered["kind"].str.contains("Case|Stay")
        ]

        enc_filtered["start"] = pd.to_datetime(enc_filtered["start"])
        enc_filtered["end"] = pd.to_datetime(enc_filtered["end"])

        # Filter encounters with duration <= 2 days
        enc_filtered = enc_filtered[
            enc_filtered["end"] - enc_filtered["start"]
            > pd.Timedelta(days=minimum_days)
        ]

        # TODO: maybe unite encounters that have one day of difference

        # Filter to keep only case encounters
        enc_filtered = enc_filtered[enc_filtered["kind"] == "Case"]

        store_df(enc_filtered, output_path, "Encounter")

    # Build BDP DataFrame
    def build_bdp_pyrate(self):
        self.default_pyrate_extraction(
            process_function=extract_bdp,
            output_name="biologically_derived_product",
            request_params=self.config["bdp_params"],
            time_attribute_name="shipStorageEnd",
        )

    def build_bdp(self):
        output_path = (
            self.config["data_dir"] / f"biologically_derived_product{OUTPUT_FORMAT}"
        )
        if self.skip_build(output_path):
            return

        bdp = self.default_metrics_extraction(
            output_name="biologically_derived_product",
            query="""
                WITH unpacked AS (
                    SELECT
                        bdp._id as metrics_id,
                        id as bdp_id,
                        bdp_rq.reference,
                        bdp_sd.start as storage_datetime,
                        bdp_sd."end" as ausgabe_datetime,
                        jsonb_array_elements(bdp.extension) as ext,
                        bdp_pro.display as display,
                        bdp_pro.code as code
                    FROM
                        biologicallyderivedproduct bdp
                    JOIN
                        biologicallyderivedproduct_request bdp_rq ON bdp_rq._resource = bdp._id
                    JOIN
                        biologicallyderivedproduct_storage_duration bdp_sd ON bdp_sd._resource = bdp._id
                    JOIN
                        "biologicallyderivedproduct_productCode_coding" bdp_pro on bdp_pro._resource = bdp._id

                ),
                filtered AS (
                    SELECT
                        metrics_id,
                        bdp_id,
                        replace(reference, 'ServiceRequest/', '') as service_request_id,
                        lower(storage_datetime) as storage_datetime,
                        lower(ausgabe_datetime) as ausgabe_datetime,
                        display,
                        code,
                        CASE
                            WHEN ext->>'url' = 'https://uk-essen.de/LAB/Nexus/Swisslab/KONSERVE/VERBRAUCH' THEN ext->>'valueString'
                            ELSE NULL
                        END as verbrauch,
                        CASE
                            WHEN ext->>'url' = 'https://uk-essen.de/LAB/Nexus/Swisslab/KONSERVE/AUSGABEAN' THEN ext->>'valueString'
                            ELSE NULL
                        END as ausgabean
                    FROM unpacked
                    WHERE ext->>'url' IN ('https://uk-essen.de/LAB/Nexus/Swisslab/KONSERVE/VERBRAUCH', 'https://uk-essen.de/LAB/Nexus/Swisslab/KONSERVE/AUSGABEAN')
                )
                SELECT
                    metrics_id,
                    bdp_id,
                    service_request_id,
                    storage_datetime,
                    ausgabe_datetime,
                    MAX(verbrauch) as verbrauch,
                    MAX(ausgabean) as ausgabean,
                    display,
                    code
                FROM filtered
                GROUP BY metrics_id, bdp_id, service_request_id, storage_datetime, ausgabe_datetime, display, code""",
            store=False,
        )

        bdp.drop_duplicates(subset=["metrics_id", "bdp_id"], inplace=True)
        bdp_filtered = bdp[
            (bdp["ausgabe_datetime"] >= self.config["start_datetime"])
            & (bdp["ausgabe_datetime"] <= self.config["end_datetime"])
        ]

        sr_ids = "', '".join(bdp_filtered["service_request_id"].unique())

        sr = self.default_metrics_extraction(
            output_name="service_request_bdp",
            query=f"""
            SELECT
                id as service_request_id,
                replace(sr_sub.reference, 'Patient/', '') as patient_id,
                fhirql_read_codes(status) as status,
                fhirql_read_codes(priority) as priority
            FROM servicerequest
            JOIN servicerequest_subject sr_sub on sr_sub._resource = servicerequest._id
            WHERE id in ('{sr_ids}')""",
            store=False,
        )

        bdp_filtered = pd.merge(bdp_filtered, sr, on="service_request_id", how="left")
        store_df(
            bdp_filtered,
            self.config["data_dir"] / f"biologically_derived_product{OUTPUT_FORMAT}",
            "BiologicallyDerivedProduct",
        )

    def build_procedure(self):
        self.default_metrics_extraction(
            output_name="procedure",
            query=f"""
            select pro.id as procedure_id,
            replace(pro_s.reference, 'Patient/', '') as patient_id,
            replace(pro_e.reference, 'Encounter/', '') as encounter_id,
            fhirql_read_codes(pro_code.code) as code,
            fhirql_read_codes(pro.status) as status,
            version,
            pro_code.display,
            lower("start")::timestamp as effectivedatetimestart_v1,
            upper("end")::timestamp as effectivedatetimeend_v1,
            lower("performedDateTime")::timestamp as effectivedatetimestart_v2,
            upper("performedDateTime")::timestamp as effectivedatetimeend_v2,
            replace(pro_loc.reference, 'Location/', '') as location_id
            from procedure pro
            join procedure_encounter pro_e on pro_e._resource = pro._id
            join procedure_subject pro_s on pro_s._resource = pro._id
            join procedure_code_coding pro_code on pro_code._resource = pro._id
            left join "procedure_performedPeriod" pro_period on pro_period._resource = pro._id
            left join procedure_location pro_loc on pro._id = pro_loc._resource
            join fhirql_codes xc1 on xc1.id = pro_code.code
            join fhirql_codes xc2 on xc2.id = pro.status
            where ('{self.config["start_datetime"]}' <= lower("start")::timestamp
            and lower("start")::timestamp <= '{self.config["end_datetime"]}')
            or ('{self.config["start_datetime"]}' <= lower("performedDateTime")::timestamp
            and lower("performedDateTime")::timestamp <= '{self.config["end_datetime"]}')
            """,
        )

    # Building Condition DataFrame
    def build_condition(self):
        self.default_metrics_extraction(
            output_name="condition",
            query=f"""
            select c.id as metrics_id,
            c._id as condition_id,
            replace(c_s.reference, 'Patient/', '') as patient_id,
            replace(c_e.reference, 'Encounter/', '') as encounter_id,
            replace(c_r.reference, 'Practitioner/', '') as practitioner_id,
            lower(c."recordedDate")::timestamp as condition_date,
            fhirql_read_codes(c_cc.code) as icd_code,
            c_cc.display as icd_display,
            c_cc.version as icd_version,
            c_ccc.code as code_diagnosis_type,
            c_ccc.display as code_diagnosis_display
            from condition c
            join condition_subject c_s on c_s._resource = c._id
            join condition_encounter c_e on c_e._resource = c._id
            left join condition_recorder c_r on c_r._resource = c._id
            left join condition_code_coding c_cc on c_cc._resource = c._id
            left join condition_category_coding c_ccc on c_ccc._resource = c._id
            where '{self.config["start_datetime"]}' <= lower(c."recordedDate")::timestamp
            and lower(c."recordedDate")::timestamp <= '{self.config["end_datetime"]}'
            """,
        )

    def build_meta_patient_pyrate(self):
        output_path = self.config["data_dir"] / f"meta_patient{OUTPUT_FORMAT}"
        if self.skip_build(output_path):
            return
        encs = self.check_and_build_file("encounter")
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
            lambda x: x.linked_patient_id
            if not pd.isnull(x.linked_patient_id)
            else x.patient_id,
            axis=1,
        )

        store_df(grouped_patients, output_path, "MetaPatient")

    def build_observation(self):
        # TODO: All the other values and component are missing
        # todo be aware that an observation can also be a clinical impression ->
        """ "partOf": [
        {
            "reference": "ClinicalImpression/33dc847fb17d023ffae62eb5888c721fe092fd2df08f13580241531cfc875210"
        }
        ]"""
        self.default_metrics_extraction(
            output_name="observation",
            query=f"""
            select obs.id as metrics_id,
            obs.id as observation_id,
            lower("effectiveDateTime")::timestamp as effectiveDateTime,
            issued,
            replace(obs_sub.reference, 'Patient/', '') as patient_id,
            replace(obs_enc.reference, 'Encounter/', '') as encounter_id,
            obs_code.code as code,
            obs_code.display as display,
            obs_vq.value as value_quantity,
            obs_vq.unit as value_unit,
            obs."valueBoolean" as value_boolean,
            obs."valueDateTime" as value_datetime,
            obs."valueInteger" as value_integer,
            obs."valueString" as value_string
            from observation as obs
            join observation_code_coding obs_code on obs._id = obs_code._resource
            join observation_subject obs_sub on obs._id = obs_sub._resource
            join observation_encounter obs_enc on obs._id = obs_enc._resource
            left join "observation_valueQuantity" as obs_vq on obs._id = obs_vq._resource
            left join "observation_valueRatio" as obs_vr on obs._id = obs_vr._resource
            left join "observation_valuePeriod" as obs_per on obs._id = obs_per._resource
            where ('{self.config["start_datetime"]}' <= issued::timestamp
            and issued::timestamp <= '{self.config["end_datetime"]}')
            or ('{self.config["start_datetime"]}' <= lower("effectiveDateTime")::timestamp
            and lower("effectiveDateTime")::timestamp <= '{self.config["end_datetime"]}')
            """,
        )

    def build_observation_pyrate(self):
        self.default_pyrate_extraction(
            output_name="observation",
            time_attribute_name="date",
            request_params={"_sort": "-date"},
            fhir_paths=[
                "id",
                "effectiveDateTime",
                ("patient_id", "subject.reference.replace('Patient/', '')"),
                ("encounter_id", "encounter.reference.replace('Encounter/', '')"),
                ("code", "code.coding.code"),
                ("code_display", "code.coding.display"),
                ("value_quantity_value", "valueQuantity.value"),
                ("value_quantity_unit", "valueQuantity.unit"),
                ("codable_concept_code", "valueCodeableConcept.coding.code"),
                ("codable_concept_display", "valueCodeableConcept.coding.display"),
                ("codable_concept_text", "valueCodeableConcept.text"),
                ("component_display", "component.code.coding.display"),
                ("component_code", "component.code.coding.code"),
                ("component_string", "component.valueString"),
                ("component_value", "component.valueQuantity.value"),
            ],
            disable_parallel=True,
        )

    def build_imaging_study(self):
        self.default_metrics_extraction(
            output_name="imaging_study",
            query=f"""
            SELECT
              id AS imaging_study_id,
              fhirql_read_codes(status) as status,
              replace(replace(jsonb_path_query(_json, '$.subject.reference')::text, 'Patient/', ''), '"', '') AS "patient_id",
              replace(replace(jsonb_path_query(_json, '$.encounter.reference')::text, 'Encounter/', ''), '"', '') AS "encounter_id",
              replace(replace(jsonb_path_query(_json, '$.basedOn.reference')::text, 'ServiceRequest/', ''), '"', '') AS "service_request_id",
              description AS description,
              lower(started)::timestamp AS started,
              "numberOfSeries" AS number_series,
              "numberOfInstances" AS number_instances,
              jsonb_path_query_array(_json, '$.identifier ? (@.system == "urn:dicom:uid").value') as study_instance_uid,
              jsonb_path_query_array(_json, '$.modality.code') as modality_code,
              jsonb_path_query_array(_json, '$.modality.version') as modality_version,
              jsonb_path_query_array(_json, '$.procedureCode.coding.version') as procedure_version,
              jsonb_path_query_array(_json, '$.procedureCode.coding.display') as procedure_display,
              jsonb_path_query_array(_json, '$.procedureCode.coding.code') as procedure_code,
              jsonb_path_query_array(_json, '$.reasonCode.coding.version') as reason_version,
              jsonb_path_query_array(_json, '$.reasonCode.coding.display') as reason_display,
              jsonb_path_query_array(_json, '$.series.uid') as series_instance_uid,
              jsonb_path_query_array(_json, '$.series.numberOfInstances') as series_instances,
              jsonb_path_query_array(_json, '$.series.number') as series_number,
              jsonb_path_query_array(_json, '$.series.description') as series_description,
              jsonb_path_query_array(_json, '$.series.modality.version') as series_modality_version,
              jsonb_path_query_array(_json, '$.series.modality.code') as series_modality,
              jsonb_path_query_array(_json, '$.series.bodySite.code') as body_site
            from imagingstudy
            where '{self.config["start_datetime"]}' <= lower(started)::timestamp
            and lower(started)::timestamp <= '{self.config["end_datetime"]}'
            """,
        )

    def build_imaging_study_pyrate(self):
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
        self.default_metrics_extraction(
            output_name="diagnostic_report",
            query=f"""
            SELECT
              dr.id AS "diagnostic_report_id",
              lower(dr."effectiveDateTime") AS "effective_datetime",
              dr.issued AS "issued",
              replace(replace(jsonb_path_query(_json, '$.subject.reference')::text, 'Patient/', ''), '"', '') AS "patient_id",
              replace(replace(jsonb_path_query(_json, '$.encounter.reference')::text, 'Encounter/', ''), '"', '') AS "encounter_id",
              replace(replace(jsonb_path_query(_json, '$.imagingStudy.reference')::text, 'ImagingStudy/', ''), '"', '') AS "imaging_study_id",
              jsonb_path_query_array(_json, '$.category.coding.code') AS category,
              jsonb_path_query_array(_json, '$.category.coding.display') AS category_display,
              jsonb_path_query_array(_json, '$.presentedForm.title') AS title,
              jsonb_path_query_array(_json, '$.presentedForm.data') AS data,
              jsonb_path_query_array(_json, '$.presentedForm.contentType') AS content_type,
              jsonb_path_query_array(_json, '$.presentedForm.url') AS url
            FROM diagnosticreport dr
            WHERE ('{self.config["start_datetime"]}' <= LOWER(dr."effectiveDateTime")::timestamp
            AND LOWER(dr."effectiveDateTime") <= '{self.config["end_datetime"]}')
            OR ('{self.config["start_datetime"]}' <= dr.issued::timestamp
            AND dr.issued <= '{self.config["end_datetime"]}');
            """,
            # timestamp_columns=["issued", "effectiveDateTime"],
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
            category = next(
                iter(c for c in row_dict["category_display"] or [] if " " not in c),
                "unknown",
            )
            document_path = (
                document_folder / category / f"{row_dict['diagnostic_report_id']}.txt"
            )
            if document_path.exists():
                continue
            if row_dict["data"] is not None and len(row_dict["data"]) > 0:
                assert len(row_dict["data"]) == 1, row_dict["diagnostic_report_id"]
                txt = base64.b64decode(row_dict["data"][0]).decode()
            elif row_dict["url"] is not None and len(row_dict["url"]) > 0:
                for choice in ["text", "pdf", "word"]:
                    chosen_id = next(
                        iter(
                            i
                            for i in range(len(row_dict["content_type"])) or []
                            if choice in row_dict["content_type"][i]
                        ),
                        None,
                    )
                    if chosen_id is not None:
                        break
                if chosen_id is not None:
                    txt = get_text(
                        ume_auth.session,
                        row_dict["url"][chosen_id],
                        content=row_dict["content_type"][chosen_id],
                    )
                else:
                    failed.append(row_dict)
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
        self.default_metrics_extraction(
            output_name="episode_of_care",
            query=f"""
            select eoc._id as metrics_id,
            jsonb_path_query(eoc.extension, '$ ? (@.url == "https://uk-essen.de/HIS/Cerner/Medico/TumorDocumentation/Onkozert").valueString') as cert,
            jsonb_path_query(eoc.extension, '$ ? (@.url == "https://uk-essen.de/HIS/Cerner/Medico/TumorDocumentation/Primärfall").valueBoolean') as primary_case,
            jsonb_path_query(eoc.extension, '$ ? (@.url == "https://uk-essen.de/HIS/Cerner/Medico/TumorDocumentation/Primärfalljahr").valueString') as primary_year,
            jsonb_path_query(eoc.extension, '$ ? (@.url == "http://uk-koeln.de/fhir/StructureDefinition/Extension/nNGM/Erstdiagnose").valueDateTime') as first_diagnosis_date,
            jsonb_path_query(eoc.extension, '$.** ? (@.url == "https://uk-essen.de/HIS/Cerner/Medico/TumorDocumentation/TumorIdentifier/Bogen").valueString') as treatment_program,
            replace(eoc_pat.reference, 'Patient/', '') as patient_id,
            lower(start) as start,
            upper("end") as end
            from episodeofcare eoc
            join episodeofcare_patient eoc_pat on eoc_pat._resource = eoc._id
            left join episodeofcare_period eoc_p on eoc_p._resource = eoc._id
            where '{self.config["start_datetime"]}'::timestamp <= lower(start)::timestamp
            and lower(start)::timestamp <= '{self.config["end_datetime"]}'::timestamp
            """,
        )

    def build_service_request_pyrate(self):
        self.default_pyrate_extraction(
            output_name="service_request",
            time_attribute_name="authored",
            request_params=self.config["service_request_params"],
        )

    def build_service_request(self):
        self.large_metrics_extraction(
            output_name="service_request",
            query_template="""
            SELECT
              id AS service_request_id,
              fhirql_read_codes(status) AS status,
              replace(replace(jsonb_path_query(_json, '$.subject.reference')::text, 'Patient/', ''), '"', '') AS "patient_id",
              replace(replace(jsonb_path_query(_json, '$.basedOn.reference')::text, 'ServiceRequest/', ''), '"', '') AS "service_request_id",
              fhirql_read_codes(intent) as intent,
              fhirql_read_codes(priority) as priority,
              jsonb_path_query_array(_json, '$.code.coding.code') AS code,
              jsonb_path_query_array(_json, '$.code.coding.display') AS code_display,
              jsonb_path_query_array(_json, '$.category.coding.code') AS category_code,
              jsonb_path_query_array(_json, '$.category.coding.display') AS category_display,
              lower("authoredOn") AS authored
            FROM servicerequest
            where "authoredOn" && tstzrange('{}'::date, '{}'::date, '[)')""",
        )

    def build_medication(self):
        self.large_metrics_extraction(
            output_name="medication",
            query_template="""
            select
                md.id as medication_id,
                md_ad.id as medication_administration_id,
                replace(md_subj.reference, 'Patient/', '') as patient_id,
                replace(md_ct.reference, 'Encounter/', '') as encounter_id,
                text as medication,
                lower("effectiveDateTime")::timestamp as datetime,
                fhirql_read_codes(md_ad.status) as status
            from
                medication md
            join
                medication_code md_cod on md_cod._resource = md._id
            join
                medicationadministration md_ad
                join "medicationadministration_medicationReference" md_ref on md_ref._resource = md_ad._id
            on
                replace(md_ref.reference, 'Medication/', '') = md.id
            join medicationadministration_subject md_subj on md_subj._resource = md_ad._id
            join medicationadministration_context md_ct on md_ct._resource = md_ad._id
            where "effectiveDateTime" && tstzrange('{}'::date, '{}'::date, '[)')""",
        )
