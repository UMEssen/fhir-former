# FHIRFormer

## How to run

```
poetry install
```

```
poetry run fhirformer --task [pretrain|ds_icd|ds_image|ds_main_diag]
```


## Run with docker

```
docker compose run trainer
```

and inside the docker container
```
python -m fhirformer --task [pretrain|ds_icd|ds_image|ds_main_diag]
```
