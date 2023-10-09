import logging
import traceback
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime

logger = logging.getLogger(__name__)


def make_timezone_aware(value, default_timezone="UTC", errors="coerce"):
    if isinstance(value, pd.Series):
        # Convert to datetime Series and ensure it's timezone-aware
        datetime_series = pd.to_datetime(value, utc=True, errors=errors).dt.tz_convert(
            default_timezone
        )
        return datetime_series
    elif isinstance(value, pd.Timestamp):
        # Check if the Timestamp is timezone-aware‚‚
        if not value.tz:
            return value.tz_localize(default_timezone)
        return value
    elif isinstance(value, str):
        timestamp = pd.to_datetime(value, errors=errors)
        if not timestamp.tz:
            return timestamp.tz_localize(default_timezone)
        return timestamp
    else:
        raise ValueError(
            f"Unsupported type {type(value)}. Expecting a pd.Series, pd.Timestamp, or str."
        )


def select_resources(
    resource_df: pd.DataFrame,
    column: str,
    patient_id: str,
    start_filter_date=None,
    end_filter_date=None,
    start_inclusive: bool = True,
    end_inclusive: bool = False,
):
    if len(resource_df) == 0:
        return resource_df

    try:
        resource_for_patient = resource_df.patient_id == patient_id
        resource_tz_column = make_timezone_aware(resource_df[column])

        if start_filter_date:
            start_filter_date = make_timezone_aware(start_filter_date)
            resource_for_patient &= (
                resource_tz_column >= start_filter_date
                if start_inclusive
                else resource_tz_column > start_filter_date
            )

        if end_filter_date:
            end_filter_date = make_timezone_aware(end_filter_date)
            resource_for_patient &= (
                resource_tz_column <= end_filter_date
                if end_inclusive
                else resource_tz_column < end_filter_date
            )
        return resource_df.loc[resource_for_patient]

    except OutOfBoundsDatetime:
        traceback.print_exc()
        logger.error(
            f"patient_id: {patient_id}, "
            f"column: {column}, "
            f"start_filter_date: {start_filter_date}, "
            f"end_filter_date: {start_filter_date}, "
            f"resource_df: \n{resource_df}"
        )
        raise OutOfBoundsDatetime


@dataclass
class DataStore:
    patient_df: pd.DataFrame
    patient_list: List[str]
    resources: Dict[str, pd.DataFrame]
    date_columns: Dict[str, str]

    def filter_patient(
        self,
        patient_id: str,
        start_filter_date=None,
        end_filter_date=None,
        target_resource=None,
        end_inclusive: bool = False,
        start_inclusive: bool = True,
    ):
        filtered_patient = self.patient_df[self.patient_df.patient_id == patient_id]

        if target_resource:
            if target_resource not in self.resources:
                raise ValueError(
                    f"Resource {target_resource} not found in available resources."
                )

            filtered_resources = {
                target_resource: select_resources(
                    resource_df=self.resources[target_resource],
                    column=self.date_columns[target_resource],
                    patient_id=patient_id,
                    start_filter_date=start_filter_date,
                    end_filter_date=end_filter_date,
                    start_inclusive=start_inclusive,
                    end_inclusive=end_inclusive,
                )
            }
        else:
            filtered_resources = {
                resource_name: select_resources(
                    resource_df=resource_df,
                    column=self.date_columns[resource_name],
                    patient_id=patient_id,
                    start_filter_date=start_filter_date,
                    end_filter_date=end_filter_date,
                    start_inclusive=start_inclusive,
                    end_inclusive=end_inclusive,
                )
                for resource_name, resource_df in self.resources.items()
            }

        return DataStore(
            patient_df=filtered_patient,
            patient_list=filtered_patient["patient_id"].unique().tolist(),
            resources=filtered_resources,
            date_columns=self.date_columns,
        )
