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
CUDA_VISIBLE_DEVICES=0,1,2 \
    torchrun --nproc_per_node 3 -m \
    fhirformer --task [pretrain_fhir|pretrain_documents|pretrain_fhir_documents|ds_icd|ds_image|ds_main_diag]
```
