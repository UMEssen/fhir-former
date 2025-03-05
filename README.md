# FHIRFormer

FHIRFormer is a transformer-based model for processing and analyzing FHIR (Fast Healthcare Interoperability Resources) data. It provides tools for pretraining models on FHIR data and documents, as well as downstream tasks like ICD coding, image analysis, and main diagnosis prediction.

## Features

- Pretraining on FHIR resources
- Pretraining on clinical documents
- Combined pretraining on FHIR and documents
- Downstream tasks:
  - ICD coding
  - Medical image analysis
  - Main diagnosis prediction

## Installation

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/Wizzzard93/fhirformer.git
cd fhirformer

# Install with Poetry
poetry install
```

### Using Pip

```bash
# Clone the repository
git clone https://github.com/Wizzzard93/fhirformer.git
cd fhirformer

# Install with pip
pip install -e .
```

## Usage

### Command Line Interface

```bash
# Using Poetry
poetry run fhirformer --task [task_name]

# If installed with pip
fhirformer --task [task_name]
```

Available tasks:
- `pretrain_fhir`: Pretrain on FHIR resources
- `pretrain_documents`: Pretrain on clinical documents
- `pretrain_fhir_documents`: Pretrain on both FHIR and documents
- `ds_icd`: Downstream task for ICD coding
- `ds_image`: Downstream task for image analysis
- `ds_main_diag`: Downstream task for main diagnosis prediction

### Docker

```bash
# Run with specific GPUs
GPUS=0,1,2 docker compose run trainer bash

# Inside the docker container
python -m fhirformer --task [task_name]
```

## Configuration

Configuration files are stored in the `fhirformer/config` directory. You can modify these files to customize the behavior of the models and training processes.

## Development

### Setup Development Environment

```bash
# Install development dependencies
poetry install --with dev

# Set up pre-commit hooks
pre-commit install
```

### Code Quality Tools

- Black for code formatting
- Flake8 for linting
- MyPy for type checking

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use FHIRFormer in your research, please cite:

```
@software{fhirformer2024,
  author = {Engelke, Merlin and Baldini, Giulia},
  title = {FHIRFormer: A Transformer-based Model for FHIR Data},
  year = {2024},
  publisher = {University Hospital Essen},
  url = {https://github.com/yourusername/fhirformer}
}
```

## Contributors

- Merlin Engelke (Merlin.Engelke@uk-essen.de)
- Giulia Baldini (Giulia.Baldini@uk-essen.de)
