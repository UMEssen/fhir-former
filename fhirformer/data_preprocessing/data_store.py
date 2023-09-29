from dataclasses import dataclass
import pandas as pd
from typing import Dict


def make_timezone_aware(value, default_timezone="UTC"):
    if isinstance(value, pd.Series):
        # Convert to datetime Series and ensure it's timezone-aware
        datetime_series = pd.to_datetime(value, utc=True).dt.tz_convert(
            default_timezone
        )
        return datetime_series
    elif isinstance(value, pd.Timestamp):
        # Check if the Timestamp is timezone-aware‚‚
        if not value.tz:
            return value.tz_localize(default_timezone)
        return value
    elif isinstance(value, str):
        timestamp = pd.to_datetime(value)
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
):
    if len(resource_df) == 0:
        return resource_df

    condition = resource_df.patient_id == patient_id
    resource_tz_column = make_timezone_aware(resource_df[column])

    if start_filter_date:
        start_filter_date = make_timezone_aware(start_filter_date)
        condition &= resource_tz_column >= start_filter_date

    if end_filter_date:
        end_filter_date = make_timezone_aware(end_filter_date)
        condition &= resource_tz_column <= end_filter_date

    return resource_df.loc[condition]


@dataclass
class DataStore:
    patient_df: pd.DataFrame
    resources: Dict[str, pd.DataFrame]
    date_columns: Dict[str, str]

    def filter_patient(
        self,
        patient_id: str,
        start_filter_date=None,
        end_filter_date=None,
        target_resource=None,
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
                )
                for resource_name, resource_df in self.resources.items()
            }

        return DataStore(
            patient_df=filtered_patient,
            resources=filtered_resources,
            date_columns=self.date_columns,
        )
