from typing import Dict, List

import numpy as np
import pandas as pd
from fhir_pyrate.util import FHIRObj


def get_obj(obj: FHIRObj, attr: str):
    if obj is None:
        return None
    elif isinstance(obj, List):
        if len(obj) == 0:
            return None
        elif len(obj) == 1:
            return get_obj(obj[0], attr)
        else:
            return [get_obj(o, attr) for o in obj or []]
    elif isinstance(obj, Dict):
        return obj.get(attr)
    elif isinstance(obj, FHIRObj):
        return getattr(obj, attr)
    elif isinstance(obj, str):
        return obj
    else:
        raise ValueError(f"Unknown type {type(obj)}")


# TODO: It is much easier with fhirpath, but there is currently a problem with the missing values
#  when handling the series.
def extract_imaging_study(bundle: FHIRObj) -> List[Dict]:
    records = []
    for entry in bundle.entry or []:
        resource = entry.resource
        identifier = next(
            iter(
                ide.value
                for ide in resource.identifier
                if ide.system == "urn:dicom:uid"
            ),
            None,
        )
        encounters = (
            get_obj(resource.encounter, "reference").replace("Encounter/", "")
            if resource.encounter is not None
            else None
        )
        service_request = (
            get_obj(resource.basedOn, "reference").replace("ServiceRequest/", "")
            if resource.encounter is not None
            else None
        )
        base_dict = {
            "id": resource.id,
            "status": resource.status,
            "study_instance_uid": identifier,
            "patient_id": resource.subject.reference.replace("Patient/", ""),
            "encounter_id": encounters,
            "service_request_id": service_request,
            "description": resource.description,
            "started": resource.started,
            "number_series": resource.numberOfSeries,
            "number_instances": resource.numberOfInstances,
            "modality_code": get_obj(resource.modality, "code"),
            "modality_version": get_obj(resource.modality, "version"),
            "procedure_code_version": get_obj(
                get_obj(resource.procedureCode, "coding"), "version"
            ),
            "procedure_code_display": get_obj(
                get_obj(resource.procedureCode, "coding"), "display"
            ),
            "procedure_code": get_obj(
                get_obj(resource.procedureCode, "coding"), "code"
            ),
            "reason_code_display": get_obj(
                get_obj(resource.reasonCode, "coding"), "display"
            ),
            "reason_code_version": get_obj(
                get_obj(resource.reasonCode, "coding"), "version"
            ),
        }
        for series in resource.series or []:
            series_dict = base_dict.copy()
            series_dict.update(
                {
                    "series_instance_uid": series.uid,
                    "series_number": series.number,
                    "series_instances": series.numberOfInstances,
                    "series_description": series.description,
                    "series_modality_version": get_obj(series.modality, "version"),
                    "series_modality": get_obj(series.modality, "code"),
                    "series_body_site": get_obj(series.bodySite, "code"),
                }
            )
            records.append(series_dict)

    return records


def extract_bdp(bundle):
    records = []
    for entry in bundle.entry or []:
        resource = entry.resource
        # ResourceType: BDP
        if resource.resourceType == "BiologicallyDerivedProduct":
            try:
                resource_id = resource.id
            except Exception:
                resource_id = np.nan

            try:
                request_id = resource.request[0].reference.split("ServiceRequest/")[-1]
            except Exception:
                request_id = np.nan

            try:
                ausgabe_datetime = resource.storage[0].duration.end
            except Exception:
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
            except Exception:
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
            except Exception:
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
                display = next(
                    (
                        e.display
                        for e in product_code
                        if e.system
                        == "https://uk-essen.de/LAB/Nexus/Swisslab/KONSERVE/KONSART"
                    ),
                    None,
                )
            except Exception:
                code = np.nan

            elements = {
                "resource_type": "bdp",
                "resource_id": resource_id,
                "request_id": request_id,
                "ausgabe_datetime": ausgabe_datetime,
                "ausgabe_type": ausgabe_type,
                "code": code,
                "display": display,
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
            except Exception:
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
