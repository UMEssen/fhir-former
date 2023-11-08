# FHIRFormer

## How to run

```
poetry install
```

```
poetry run fhirformer --task [pretrain_fhir|pretrain_documents|pretrain_fhir_documents|ds_icd|ds_image|ds_main_diag]
```


## Run with docker

```
GPUS=0,1,2 docker compose run trainer bash
```

and inside the docker container
```
python -m fhirformer --task [pretrain_fhir|pretrain_documents|pretrain_fhir_documents|ds_icd|ds_image|ds_main_diag]
```

distributed training
```
accelerate config --config_file conf.yaml
accelerate launch --config_file conf.yaml -m \
    fhirformer --task [pretrain_fhir|pretrain_documents|pretrain_fhir_documents|ds_icd|ds_image|ds_main_diag]
```
