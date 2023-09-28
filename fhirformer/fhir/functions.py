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
    def extract_bdp_elements(resource):
        # Extract BiologicallyDerivedProduct elements
        resource_id = getattr(resource, "id", np.nan)
        request_id = (
            getattr(resource, "request", [{}])[0]
            .get("reference", "")
            .split("ServiceRequest/")[-1]
        )
        ausgabe_datetime = (
            getattr(resource, "storage", [{}])[0].get("duration", {}).get("end", pd.NaT)
        )

        extensions = getattr(resource, "extension", [])
        output_to = next(
            (
                e.valueString
                for e in extensions
                if e.url == "https://uk-essen.de/LAB/Nexus/Swisslab/KONSERVE/AUSGABEAN"
            ),
            np.nan,
        )
        ausgabe_type = next(
            (
                e.valueString
                for e in extensions
                if e.url == "https://uk-essen.de/LAB/Nexus/Swisslab/KONSERVE/VERBRAUCH"
            ),
            np.nan,
        )

        product_code = getattr(resource, "productCode", {}).get("coding", [])
        code = next(
            (
                e.get("code")
                for e in product_code
                if e.get("system")
                == "https://uk-essen.de/LAB/Nexus/Swisslab/KONSERVE/KONSART"
            ),
            np.nan,
        )
        display = next(
            (
                e.get("display")
                for e in product_code
                if e.get("system")
                == "https://uk-essen.de/LAB/Nexus/Swisslab/KONSERVE/KONSART"
            ),
            np.nan,
        )

        return {
            "resource_type": "bdp",
            "resource_id": resource_id,
            "request_id": request_id,
            "ausgabe_datetime": ausgabe_datetime,
            "ausgabe_type": ausgabe_type,
            "code": code,
            "display": display,
            "output_to": output_to,
        }

    def extract_service_request_elements(resource):
        request_id = getattr(resource, "id", np.nan)
        patient_id = (
            getattr(resource, "subject", {}).get("reference", "").split("Patient/")[-1]
        )

        requester = getattr(resource, "requester", {})
        output_to_einskurz = requester.get("extension", [{}])[0].get("valueString")
        output_to_einscode = requester.get("extension", [{}])[1].get("valueString")

        return {
            "resource_type": "sr",
            "request_id": request_id,
            "patient_id": patient_id,
            "output_to_einskurz": output_to_einskurz,
            "output_to_einscode": output_to_einscode,
        }

    records = []

    for entry in bundle.entry or []:
        resource = entry.resource

        if resource.resourceType == "BiologicallyDerivedProduct":
            elements = extract_bdp_elements(resource)
            records.append(elements)

        if resource.resourceType == "ServiceRequest":
            elements = extract_service_request_elements(resource)
            records.append(elements)

    return records
