# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-10-01

### Changed
- Depend on `fhir-pyrate` from PyPI instead of the git `main` branch
- Require Python 3.10 or newer (needed by `fhir-pyrate` 0.2.6)
- Single build configuration in `pyproject.toml` (removed `setup.py` and `MANIFEST.in`)

### Removed
- Unused `fhir` dependency, which shadowed the `fhir` namespace used by `fhir.resources`

### Added
- GitHub Actions workflow that publishes to PyPI when a release is published

## [0.1.0] - 2024-03-05

### Added
- Proper package structure
- Setup.py for pip installation
- Comprehensive README
- MIT License
- MANIFEST.in for package data

### Changed
- Improved documentation
- Updated dependency specifications

## [0.1.0] - Initial Release

### Added
- Initial implementation of FHIRFormer
- Support for pretraining on FHIR resources
- Support for pretraining on clinical documents
- Downstream tasks for ICD coding, image analysis, and main diagnosis prediction
