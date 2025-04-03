import datetime as dt
import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, List
from urllib.parse import parse_qs, urlparse

import requests
from fhir.resources import meta, riskassessment
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.period import Period
from fhir.resources.reference import Reference
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from fhirformer.ml import inference

logger = logging.getLogger(__name__)

# FHIR Server Configuration from .envrc
SEARCH_URL = os.getenv("SEARCH_URL", "https://ship.ume.de/app/FHIR/r4")
BASIC_AUTH = os.getenv("BASIC_AUTH", "https://ship.ume.de/app/Auth/v1/basicAuth")
REFRESH_AUTH = os.getenv("REFRESH_AUTH", "https://ship.ume.de/app/Auth/v1/refresh")
FHIR_USER = os.getenv("FHIR_USER", "")
FHIR_PASSWORD = os.getenv("FHIR_PASSWORD", "")

# Configure retry strategy
retry_strategy = Retry(
    total=5,
    backoff_factor=0.1,
    status_forcelist=[500, 502, 503, 504],
)
http_adapter = HTTPAdapter(max_retries=retry_strategy)


def custom_json_serializer(obj: object) -> str:
    """Custom JSON serializer for datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat(timespec="milliseconds")
    raise TypeError(f"Type {type(obj)} not JSON serializable")


def get_time_zone_format(time: datetime) -> datetime:
    """Convert datetime to FHIR-compatible timezone format."""
    return time


def fhir_login() -> str:
    """Login to FHIR server and get authentication token."""
    try:
        session = requests.Session()
        session.mount("http://", http_adapter)
        session.mount("https://", http_adapter)

        # Only attempt auth if credentials are provided
        if FHIR_USER and FHIR_PASSWORD:
            response = session.post(
                BASIC_AUTH,
                auth=(FHIR_USER, FHIR_PASSWORD),
            )
        else:
            raise Exception("FHIR credentials not provided")

        response.raise_for_status()

        # Check for login failure
        query_args = parse_qs(urlparse(response.url).query)
        if "result" in query_args and query_args["result"][0] == "loginFailed":
            raise Exception("Invalid login credentials")

        return response.text
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to login to FHIR server: {str(e)}")
        raise


class FHIRPredictor:
    def __init__(self, config: Dict, token: str):
        self.token = token
        self.config = config
        self.session = requests.Session()
        self.session.mount("http://", http_adapter)
        self.session.mount("https://", http_adapter)
        self.meta_info = meta.Meta(
            source="urn:fhirformer:ml",
            _source=None,
            profile=["https://fhirformer.org/StructureDefinition/ml-prediction"],
            _profile=None,
            extension=[
                {
                    "url": "http://fhirformer.org/extension-source-version",
                    "valueString": "0.0.1",
                }
            ],
            fhir_comments=None,
            id=None,
            lastUpdated=None,
            _lastUpdated=None,
            security=[],
            tag=[],
            versionId=None,
            _versionId=None,
        )
        # Task-specific configurations
        self.task_configs = {
            "ds_image": {
                "system": "https://ship.ume.de/fhir/ml/imaging-modality",
                "display_prefix": "Predicted Imaging Modality",
                "resource_type": "ImagingStudy",
            },
            "ds_icd": {
                "system": "https://ship.ume.de/fhir/ml/icd-code",
                "display_prefix": "Predicted ICD Code",
                "resource_type": "Condition",
            },
            "ds_readmission": {
                "system": "https://ship.ume.de/fhir/ml/readmission",
                "display_prefix": "Predicted Readmission Risk",
                "resource_type": "Encounter",
            },
            "ds_mortality": {
                "system": "https://ship.ume.de/fhir/ml/mortality",
                "display_prefix": "Predicted Mortality Risk",
                "resource_type": "Encounter",
            },
        }
        task = config.get("task", "ds_image")
        self.task_config = self.task_configs.get(
            task,
            {
                "system": "https://ship.ume.de/fhir/ml/unknown",
                "display_prefix": "Prediction",
                "resource_type": "Observation",
            },
        )
        logger.info(f"Using task configuration for: {task}")

    def create_risk_assessment(
        self,
        patient_id: str,
        prediction: str,
        probability: float,
        text: str,
        basis_resources: List[Reference],
        model_name: str,
    ) -> riskassessment.RiskAssessment:
        """Create a FHIR RiskAssessment resource."""
        # Generate unique ID for the risk assessment
        risk_id = hashlib.sha256(
            "".join(
                [
                    "fhirformer",
                    patient_id,
                    model_name,
                    str(dt.datetime.now()),
                ]
            ).encode()
        ).hexdigest()

        # Current time in UTC
        now = datetime.now(tz=timezone.utc)

        # Create RiskAssessment resource
        risk_assessment = riskassessment.RiskAssessment(
            id=risk_id,
            meta=self.meta_info,
            status="final",
            subject=Reference(
                reference=f"Patient/{patient_id}",
                display=f"Patient {patient_id}",
                type="Patient",
                identifier=None,
                extension=[],
                id=None,
                fhir_comments=None,
                _display=None,
                _reference=None,
                _type=None,
            ),
            occurrenceDateTime=now,
            method=CodeableConcept(
                coding=[
                    Coding(
                        code=model_name,
                        system="https://ship.ume.de/fhir/ml/models",
                        display=f"FHIRFormer ML Model: {model_name}",
                        version="1.0",
                        userSelected=None,
                        extension=[],
                        id=None,
                        fhir_comments=None,
                        _code=None,
                        _display=None,
                        _system=None,
                        _userSelected=None,
                        _version=None,
                    )
                ],
                text=f"ML Model: {model_name}",
                extension=[],
                id=None,
                fhir_comments=None,
                _text=None,
            ),
            basis=basis_resources,
            note=[{"text": f"Input text: {text}"}],
            prediction=[
                {
                    "probabilityDecimal": float(probability),
                    "whenPeriod": Period(
                        start=now,
                        end=now + dt.timedelta(days=1),
                        extension=[],
                        id=None,
                        fhir_comments=None,
                        _end=None,
                        _start=None,
                    ),
                    "qualitativeRisk": CodeableConcept(
                        coding=[
                            Coding(
                                system=self.task_config["system"],
                                code=str(prediction),
                                display=f"{self.task_config['display_prefix']}: {str(prediction)}",
                                version="1.0",
                                userSelected=None,
                                extension=[],
                                id=None,
                                fhir_comments=None,
                                _code=None,
                                _display=None,
                                _system=None,
                                _userSelected=None,
                                _version=None,
                            )
                        ],
                        text=f"{self.task_config['display_prefix']}: {str(prediction)}",
                        extension=[],
                        id=None,
                        fhir_comments=None,
                        _text=None,
                    ),
                }
            ],
            language="en",
            implicitRules=None,
            _implicitRules=None,
            _language=None,
            _mitigation=None,
            _occurrenceDateTime=None,
            occurrencePeriod=None,
            _status=None,
            contained=[],
            extension=[],
            modifierExtension=[],
            text=None,
            basedOn=None,
            code=None,
            condition=None,
            encounter=None,
            identifier=[],
            mitigation=None,
            parent=None,
            performer=None,
            reason=[],
            fhir_comments=None,
        )

        return risk_assessment

    def push_to_fhir(self, risk_assessment: riskassessment.RiskAssessment) -> bool:
        """Push RiskAssessment resource to FHIR server."""
        try:
            response = self.session.put(
                f"{SEARCH_URL}/RiskAssessment/{risk_assessment.id}",
                json=json.loads(
                    json.dumps(
                        risk_assessment.dict(),
                        default=custom_json_serializer,
                    )
                ),
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Content-Type": "application/fhir+json",
                },
            )
            response.raise_for_status()
            logger.info(f"Successfully created RiskAssessment {risk_assessment.id}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to push RiskAssessment: {str(e)}")
            return False

    def get_basis_resources(
        self, text: str, patient_id: str, encounter_id: str, sample_start: str
    ) -> List[Reference]:
        """Get list of resources used as basis for prediction."""
        basis = []

        # Add encounter reference
        basis.append(
            Reference(
                reference=f"Encounter/{encounter_id}",
                display=f"Encounter for patient {patient_id}",
                type="Encounter",
                identifier=None,
                extension=[],
                id=None,
                fhir_comments=None,
                _display=None,
                _reference=None,
                _type=None,
            )
        )

        # Extract resources from history based on task type
        history_started = False
        for line in text.split("\n"):
            if line.startswith("Patient history:"):
                history_started = True
                continue
            if history_started and line.strip():
                # Parse date and study
                parts = line.split(":", 1)
                if len(parts) == 2:
                    date = parts[0]
                    resource_data = parts[1].strip()
                    # Only include resources before sample_start
                    if date <= sample_start and resource_data:
                        # Create a reference for each resource
                        resource_id = hashlib.sha256(
                            f"{patient_id}:{date}:{resource_data}".encode()
                        ).hexdigest()
                        basis.append(
                            Reference(
                                reference=f"{self.task_config['resource_type']}/{resource_id}",
                                display=f"{resource_data} on {date}",
                                type=self.task_config["resource_type"],
                                identifier=None,
                                extension=[],
                                id=None,
                                fhir_comments=None,
                                _display=None,
                                _reference=None,
                                _type=None,
                            )
                        )

        return basis

    def process_predictions(self, predictions_data: Dict) -> None:
        """Process predictions and create FHIR resources."""
        if not predictions_data:
            logger.error("No predictions data provided")
            return

        texts = predictions_data.get("texts", [])
        predictions = predictions_data.get("predictions", [])
        probabilities = predictions_data.get("probabilities", [])

        if not texts or not predictions or not probabilities:
            logger.error("Missing required prediction data")
            return

        # Get model name from config
        model_name = self.config.get("model_name", "unknown_model")

        logger.info(f"Processing {len(texts)} predictions for model {model_name}")

        # Process each prediction
        for i, (text, prediction, probability) in enumerate(
            zip(texts, predictions, probabilities)
        ):
            try:
                # Extract patient_id and other metadata from text
                patient_id = None
                encounter_id = None
                sample_start = None

                for line in text.split("\n"):
                    if "Patient metadata:" in line:
                        # Extract patient_id from metadata
                        metadata_lines = text.split("Patient metadata:\n")[1].split(
                            "\n"
                        )
                        for meta_line in metadata_lines:
                            if meta_line.startswith("patient_id"):
                                patient_id = meta_line.split("\t")[1].strip()
                    elif "Encounter:" in line:
                        # Extract encounter_id and sample_start from encounter info
                        encounter_lines = text.split("Encounter:\n")[1].split("\n")
                        for enc_line in encounter_lines:
                            if enc_line.startswith("id"):
                                encounter_id = enc_line.split("\t")[1].strip()
                            elif enc_line.startswith("Beginn"):
                                sample_start = enc_line.split("\t")[1].strip()

                if not all([patient_id, encounter_id, sample_start]):
                    logger.warning(
                        f"Missing required metadata for prediction {i + 1}, skipping. "
                        f"patient_id: {patient_id}, encounter_id: {encounter_id}, sample_start: {sample_start}"
                    )
                    continue

                # Create and push risk assessment
                risk = self.create_risk_assessment(
                    patient_id=str(patient_id),
                    prediction=str(prediction),
                    probability=float(probability),
                    text=text,
                    basis_resources=self.get_basis_resources(
                        text=text,
                        patient_id=str(patient_id),
                        encounter_id=str(encounter_id),
                        sample_start=str(sample_start),
                    ),
                    model_name=model_name,
                )

                success = self.push_to_fhir(risk)
                if success:
                    logger.info(f"Processed prediction {i + 1}/{len(texts)}")
                else:
                    logger.error(f"Failed to process prediction {i + 1}/{len(texts)}")

            except Exception as e:
                logger.error(f"Error processing prediction {i + 1}: {str(e)}")


def main(config: Dict) -> None:
    """Main function to run predictions and push to FHIR."""
    try:
        # Update model path to use artifacts directory if provided
        if config.get("model_checkpoint"):
            logger.info(f"Using model from checkpoint: {config['model_checkpoint']}")

        # Run inference using inference.py
        logger.info("Running inference to generate predictions...")
        inference_results = inference.main(
            config, num_wandb_samples=0
        )  # Don't log to wandb for live inference

        if not inference_results:
            logger.error("No inference results returned")
            return

        # Login to FHIR server
        logger.info("Logging into FHIR server...")
        token = fhir_login()
        if not token:
            logger.error("Failed to get FHIR token")
            return
        logger.info("Successfully logged into FHIR server")

        # Initialize predictor with task-specific configuration
        predictor = FHIRPredictor(config, token)

        # Process predictions and push to FHIR
        logger.info("Processing predictions and pushing to FHIR...")
        predictor.process_predictions(inference_results)

        logger.info("Completed processing predictions and pushing to FHIR")

    except Exception as e:
        logger.error(f"Error in prediction pipeline: {str(e)}")
        raise
