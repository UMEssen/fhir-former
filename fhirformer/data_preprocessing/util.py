import logging

from fhirformer.fhir.util import check_and_read


def get_valid_labels(path: str, column: str, percentual_cutoff: float = 0.005) -> list:
    resource = check_and_read(path)
    codes = resource[column].value_counts(normalize=True)
    logging.info(f"Number of unique codes: {len(codes)}")
    filtered_codes = codes[codes > percentual_cutoff].index.tolist()
    logging.info(f"Number of unique codes after filtering: {len(filtered_codes)}")
    return filtered_codes


def validate_resources(resources, config):
    for resource in resources:
        if resource not in config["text_sampling_column_maps"].keys():
            raise NotImplementedError(
                f"Resource {resource} not in config['text_sampling_column_maps'].keys()."
            )


def get_data_info(pats_int, store_list_global):
    logging.info(f"Overall patients to process {pats_int}")
    logging.info(f"{pats_int} are divided into {len(store_list_global)} lists")
    logging.info(f"Split to patient ratio {pats_int/len(store_list_global)}")


def get_column_map_txt_resources(config, resources_for_pre_training):
    return {
        k: v
        for k, v in config["text_sampling_column_maps"].items()
        if k in resources_for_pre_training
    }


def get_patient_ids_lists(store_list_global):
    return [
        store_global.patient_df["patient_id"].unique().tolist()
        for store_global in store_list_global
    ]


def is_skip_build(config: dict) -> bool:
    return (
        True
        if (
            not config["rerun_cache"]
            and (config["task_dir"] / "train.json").exists()
            and (config["task_dir"] / "test.json").exists()
            and not config["debug"]
        )
        else False
    )
