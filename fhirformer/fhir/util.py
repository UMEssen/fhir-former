import logging
import os
import pickle
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from fhir_pyrate import Ahoy
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)
metrics_url = (
    f"postgresql://{os.environ['METRICS_USER']}:{os.environ['METRICS_PASSWORD']}@"
    f"{os.environ['METRICS_HOSTNAME']}:{os.environ['METRICS_PORT']}/{os.environ['METRICS_DB']}"
)
engine = create_engine(metrics_url)
auth = Ahoy(
    auth_method="env",
    username=os.environ["FHIR_USER"],
    auth_url=os.environ["BASIC_AUTH"],
    refresh_url=os.environ["REFRESH_AUTH"],
)

OUTPUT_FORMAT = ".pkl"
FOLDER_DEPTH = 64 // 8


def group_meta_patients(group_by_tuple: Tuple[str, pd.DataFrame]) -> List[Dict]:
    _, df = group_by_tuple
    patient_ids = list(
        set(
            list(df["patient_id"].dropna().values)
            + list(df["patient_id_meta"].dropna().values)
            + list(df["other_list"].dropna().values)
        )
    )
    rows = []
    for row in ["birth_date", "deceased_date", "insurance_type", "sex"]:
        rows.append(
            list(
                set(
                    list(df[row].dropna().values)
                    + list(df[f"{row}_meta"].dropna().values)
                )
            )
        )
    birthdate, deceased_date, insurance_type, sex = rows
    # Birthdate must be the same for all patients and must exist
    if len(birthdate) > 1:
        # patients with multible birthdates are not considered
        logging.info(f"{patient_ids}: {birthdate}, more than one date found!")
        return []
    assert len(birthdate) == 1, f"{patient_ids}: {birthdate}"
    birthdate = pd.Timestamp(birthdate[0])
    if len(deceased_date) == 1:
        deceased_date = pd.Timestamp(deceased_date[0])
    elif len(deceased_date) > 1:
        b1 = pd.Timestamp(deceased_date[0])
        for b2 in deceased_date[1:]:
            offset = abs((b1 - pd.Timestamp(b2)).days)
            # If any of the dates are more than 3 days apart, we assume that this is wrong
            if offset > 3:
                print(f"{patient_ids}: {deceased_date}, more than one date found!")
                return []
        deceased_date = b1
    else:
        deceased_date = None

    if deceased_date is not None:
        assert deceased_date >= birthdate, f"{patient_ids}: {deceased_date}"
    if "male" in sex and "female" in sex:
        final_sex = "unknown"
    elif "male" in sex:
        final_sex = "male"
    elif "female" in sex:
        final_sex = "female"
    else:
        final_sex = "other"

    return [
        dict(
            patient_id=patient_id,
            linked_patient_id=",".join(sorted(patient_ids)),
            birth_date=birthdate,
            insurance_type=",".join(insurance_type),
            sex=final_sex,
            deceased_date=deceased_date,
        )
        for patient_id in patient_ids
    ]


def choose_list_items(
    x: Any, set_to_none: bool = False, take_first: bool = False
) -> Any:
    if isinstance(x, list):
        if len(x) == 0:
            return None
        elif len(x) == 1 or take_first:
            return sorted(x)[0]
        elif set_to_none:
            return None
        else:
            return x
    else:
        return x


def store_df(df: pd.DataFrame, output_path: Path, resource: str = "resource"):
    logger.info(f"Saving {len(df)} {resource} to {output_path}")
    if output_path.name.endswith(".ftr"):
        df.reset_index(drop=True).to_feather(output_path)
    elif output_path.name.endswith(".pkl"):
        df.reset_index(drop=True).to_pickle(output_path)
    else:
        raise ValueError(f"Output format not supported for {output_path}")


def load_df(input_path: Path, resource: str = "resource") -> pd.DataFrame:
    logger.info(f"Loading {resource} from {input_path}")
    if input_path.name.endswith(".ftr"):
        return pd.read_feather(input_path)
    elif input_path.name.endswith(".pkl"):
        return pd.read_pickle(input_path)
    else:
        raise ValueError(f"Input format not supported for {input_path}")


def check_and_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(
            str(path) + " does not exist, please run the FHIRExtractor first."
        )
    if str(path).endswith(".ftr"):
        return pd.read_feather(path)
    elif str(path).endswith(".pkl"):
        with path.open("rb") as of:
            return pickle.load(of)
    else:
        raise ValueError("File type of " + str(path) + " not supported.")


def get_document_path(root_path: Path, filename: str, folder_depth: int = 0):
    if folder_depth == 0:
        return root_path / filename
    else:
        hex_filename = filename.replace(".txt", "")
        # Create subfolders based on the filename
        subfolders = [
            hex_filename[i : i + folder_depth]
            for i in range(0, len(hex_filename), folder_depth)
        ]
        destination_folder = root_path / "/".join(subfolders)
        return destination_folder / filename


def get_category_name(categories: List[str]):
    if categories is None or len(categories) == 0:
        return "unknown"
    category = next(
        iter(c for c in categories or [] if " " not in c),
        None,
    )
    if category is None:
        category = categories[0].replace(" ", "_")
    return category


def get_text(
    session: requests.Session, url: str, content=None, row_for_debug=None
) -> Optional[str]:
    new_url = url if content in {None, "application/txt"} else url + "/txt"
    if new_url is None:
        return None
    try:
        response = session.get(new_url)
        return response.text
    except Exception:
        traceback.print_exc()
        logging.error(f"Debug: {row_for_debug}")
        return None


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


def extract_id(element: pd.Series) -> pd.Series:
    return element.str.split("/").str[-1]


def reduce_cardinality(
    series: pd.Series, set_to_none: bool = False, take_first: bool = False
) -> pd.Series:
    return series.apply(lambda x: choose_list_items(x, set_to_none, take_first))
