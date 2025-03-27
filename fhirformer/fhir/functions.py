from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fhir_pyrate.util import FHIRObj


def get_obj(obj: FHIRObj | Dict | List | None, attr: str) -> Any:
    if obj is None:
        return None
    elif isinstance(obj, List):
        if len(obj) == 0:
            return None
        elif len(obj) == 1:
            return get_obj(obj[0], attr)
        else:
            return [get_obj(o, attr) for o in obj]
    elif isinstance(obj, Dict):
        return obj.get(attr)
    elif isinstance(obj, FHIRObj):
        return getattr(obj, attr, None)
    elif isinstance(obj, str):
        return obj
    else:
        return None


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

        # Safely handle encounter reference
        encounter_ref = get_obj(resource.encounter, "reference")
        encounters = (
            encounter_ref.replace("Encounter/", "")
            if isinstance(encounter_ref, str)
            else None
        )

        # Safely handle service request reference
        based_on_ref = get_obj(resource.basedOn, "reference")
        service_request = (
            based_on_ref.replace("ServiceRequest/", "")
            if isinstance(based_on_ref, str)
            else None
        )

        # Safely handle subject reference
        subject_ref = get_obj(resource.subject, "reference")
        patient_id = (
            subject_ref.replace("Patient/", "")
            if isinstance(subject_ref, str)
            else None
        )

        # Helper function to safely get nested coding values
        def get_coding_value(obj: Any, value_type: str) -> Any:
            coding = get_obj(obj, "coding")
            if coding is None:
                return None
            if isinstance(coding, list):
                first_coding = coding[0] if coding else None
                return get_obj(first_coding, value_type)
            return get_obj(coding, value_type)

        base_dict = {
            "id": resource.id,
            "status": resource.status,
            "study_instance_uid": identifier,
            "patient_id": patient_id,
            "encounter_id": encounters,
            "service_request_id": service_request,
            "description": resource.description,
            "started": resource.started,
            "number_series": resource.numberOfSeries,
            "number_instances": resource.numberOfInstances,
            "modality_code": get_obj(resource.modality, "code"),
            "modality_version": get_obj(resource.modality, "version"),
            "procedure_code_version": get_coding_value(
                resource.procedureCode, "version"
            ),
            "procedure_code_display": get_coding_value(
                resource.procedureCode, "display"
            ),
            "procedure_code": get_coding_value(resource.procedureCode, "code"),
            "reason_code_display": get_coding_value(resource.reasonCode, "display"),
            "reason_code_version": get_coding_value(resource.reasonCode, "version"),
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


def _extract_bdp_extension_value(extensions, url: str) -> Any:
    """Extract value from extensions based on URL."""
    if extensions is not None and not isinstance(extensions, float):
        return next(
            (e.valueString for e in extensions if e.url == url),
            None,
        )
    return np.nan


def _extract_product_code_info(product_code) -> tuple[Any, Any]:
    """Extract code and display from product code."""
    try:
        system_url = "https://uk-essen.de/LAB/Nexus/Swisslab/KONSERVE/KONSART"
        code = next(
            (e.code for e in product_code if e.system == system_url),
            None,
        )
        display = next(
            (e.display for e in product_code if e.system == system_url),
            None,
        )
        return code, display
    except Exception:
        return np.nan, np.nan


def _extract_bdp_resource(resource) -> dict:
    """Extract information from BDP resource."""
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

    # Extract extension values
    output_to = _extract_bdp_extension_value(
        resource.extension, "https://uk-essen.de/LAB/Nexus/Swisslab/KONSERVE/AUSGABEAN"
    )

    ausgabe_type = _extract_bdp_extension_value(
        resource.extension, "https://uk-essen.de/LAB/Nexus/Swisslab/KONSERVE/VERBRAUCH"
    )

    # Extract product code information
    code, display = _extract_product_code_info(resource.productCode.coding)

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


def _extract_service_request(resource) -> dict:
    """Extract information from ServiceRequest resource."""
    request_id = resource.id
    patient_id = resource.subject.reference.split("Patient/")[-1]
    try:
        output_to_einskurz = resource.requester.extension[0].valueString
        output_to_einscode = resource.requester.extension[1].valueString
    except Exception:
        output_to_einskurz = None
        output_to_einscode = None

    return {
        "resource_type": "sr",
        "request_id": request_id,
        "patient_id": patient_id,
        "output_to_einskurz": output_to_einskurz,
        "output_to_einscode": output_to_einscode,
    }


def extract_bdp(bundle):
    """Extract BDP and ServiceRequest information from bundle."""
    records = []
    for entry in bundle.entry or []:
        resource = entry.resource

        if resource.resourceType == "BiologicallyDerivedProduct":
            records.append(_extract_bdp_resource(resource))
        elif resource.resourceType == "ServiceRequest":
            records.append(_extract_service_request(resource))

    return records
