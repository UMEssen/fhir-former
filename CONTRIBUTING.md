# Contributing to FHIRFormer

Thank you for considering contributing to FHIRFormer! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## How to Contribute

1. Fork the repository
2. Create a new branch for your feature or bugfix
3. Make your changes
4. Run tests and ensure code quality
5. Submit a pull request

## Development Environment Setup

```bash
# Clone your fork
git clone https://github.com/your-username/fhirformer.git
cd fhirformer

# Install development dependencies
poetry install --with dev

# Set up pre-commit hooks
pre-commit install
```

## Code Quality

Before submitting a pull request, please ensure your code:

- Passes all tests
- Follows the project's coding style (Black formatting)
- Passes linting with Flake8
- Passes type checking with MyPy

```bash
# Format code
black .

# Run linting
flake8

# Run type checking
mypy
```

## Pull Request Process

1. Update the README.md or documentation with details of changes if appropriate
2. Update the CHANGELOG.md with details of changes
3. The PR should work for Python 3.9 and above
4. PRs will be merged once they have been reviewed and approved

## Feature Requests and Bug Reports

Please use the GitHub issue tracker to submit feature requests and bug reports.

## Contact

If you have any questions, feel free to contact the maintainers:
- Merlin Engelke (Merlin.Engelke@uk-essen.de)
- Giulia Baldini (Giulia.Baldini@uk-essen.de)
