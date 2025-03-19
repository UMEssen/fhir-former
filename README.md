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

### Creating New Downstream Tasks

To create a new downstream task:

1. Add your task configuration in `fhirformer/config/config_training.yaml`:
```yaml
data_id: {
    // ... existing tasks ...
    "ds_your_task": "V1"  # Add your task here
}

resources_for_task: {
    "ds_your_task": [
        # List required FHIR resources for your task
        "condition",
        "procedure",
        # Add other needed resources
    ]
}
```

2. Create a new task builder class that inherits from `EncounterDatasetBuilder`:
```python
from fhirformer.data_preprocessing.encounter_dataset_builder import EncounterDatasetBuilder

class YourTaskBuilder(EncounterDatasetBuilder):
    def process_patient(self, patient_id: str, datastore: DataStore) -> List[Dict]:
        # Implement your task-specific patient processing logic
        # Must return a list of dictionaries containing:
        # - patient_id: str
        # - text: str (input text)
        # - labels: Any (task labels)
        pass

    def global_multiprocessing(self):
        # Implement multiprocessing logic if needed
        # Usually can reuse parent class implementation
        pass
```

3. Register your task in the CLI:
```python
# In fhirformer/cli.py
TASK_MAPPING = {
    // ... existing tasks ...
    "ds_your_task": YourTaskBuilder
}
```

4. Run your task:
```bash
poetry run fhirformer --task ds_your_task
```

Key considerations when creating a task:
- Define required FHIR resources in config_training.yaml
- Implement data processing logic in process_patient()
- Structure output as {patient_id, text, labels}

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
