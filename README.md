# FHIRFormer

FHIRFormer is a transformer-based model for processing and analyzing FHIR (Fast Healthcare Interoperability Resources) data. It provides tools for pretraining models on FHIR data and documents, as well as downstream tasks like ICD coding, image analysis, readmission prediction, and mortality prediction.

## Features

- Pretraining on FHIR resources
- Pretraining on clinical documents
- Combined pretraining on FHIR and documents
- Downstream tasks:
  - ICD coding
  - Medical image analysis
  - Readmission prediction
  - Mortality prediction
  - Main ICD prediction
- Live inference capabilities for FHIR server integration

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
poetry run fhirformer --task [task_name] [options]

# If installed with pip
fhirformer --task [task_name] [options]
```

Available tasks:
- `pretrain_fhir`: Pretrain on FHIR resources
- `pretrain_documents`: Pretrain on clinical documents
- `pretrain_fhir_documents`: Pretrain on both FHIR and documents
- `ds_icd`: Downstream task for ICD coding
- `ds_image`: Downstream task for image analysis
- `ds_readmission`: Downstream task for readmission prediction
- `ds_mortality`: Downstream task for mortality prediction
- `ds_main_icd`: Downstream task for main ICD prediction

Common options:
- `--root_dir`: Specify the root directory for data and outputs
- `--wandb`: Enable Weights & Biases logging
- `--model_checkpoint`: Path to trained model or huggingface model name
- `--debug`: Run in debug mode
- `--step`: Specify steps to run (data, sampling, train, test, all)
- `--max_train_samples`: Maximum number of training samples
- `--run_name`: Custom name for the run
- `--live_inference`: Enable live inference mode
- `--use_*`: Toggle specific FHIR resources (e.g., `--use_imaging_study`, `--use_condition`, etc.)

### Live Inference

FHIRFormer supports live inference from FHIR servers. When using `--live_inference`, the model will:
1. Download ongoinng encounters from FHIR
2. Generate "live" samples
3. Make predictions
4. Push predictions as RiskAssesment resource to FHIR

Example for image prediction task:
```bash
python -m fhirformer \
    --live_inference \
    --task ds_image \
    --use_imaging_study=True \
    --use_episode_of_care=True \
    --wandb_artifact="ship-ai-autopilot/fhirformer_ds_v2/model-o1u3iat3:v1"
```

This command will:
- Enable live inference mode
- Use the image analysis task
- Process imaging studies and episode of care data
- Load the specified model from Weights & Biases artifacts
- Make predicitons and send them to FHIR

> **Important**: The `--wandb_artifact` parameter is required for live inference. It specifies which trained model to use for predictions. It is cached once it is downloaded once.

### Docker

```bash
# Run with specific GPUs
GPUS=0,1,2 docker compose run trainer bash

# Inside the docker container
python -m fhirformer --task [task_name]
```

## Configuration

Configuration files are stored in the `fhirformer/config` directory. You can modify these files to customize the behavior of the models and training processes.

The main configuration file is `config_training.yaml` which contains:
- Data configurations
- Model parameters
- Training settings
- Task-specific configurations

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
pipelines = {
    // ... existing tasks ...
    "ds_your_task": {
        "generate": your_task_generator.main,
        "train": ds_single_label.main,  # or ds_multi_label.main
    }
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
- Choose appropriate training pipeline (single_label or multi_label)

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
