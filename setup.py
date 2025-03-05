#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name="fhirformer",
    version="0.2.0",
    description="A transformer-based model for FHIR data",
    author="Merlin Engelke, Giulia Baldini",
    author_email="Merlin.Engelke@uk-essen.de",
    url="https://github.com/yourusername/fhirformer",
    packages=find_packages(),
    include_package_data=True,
    license="MIT",
    python_requires=">=3.9",
    install_requires=[
        "scikit-learn>=1.3.0",
        "transformers[torch]>=4.33.0",
        "wandb>=0.15.8",
        "torch>=2.0.0,!=2.0.1",
        "datasets>=2.14.4",
        "sqlalchemy>=2.0.20",
        "pyyaml>=6.0.1",
        "python-dotenv>=1.0.0",
        "psycopg2-binary>=2.9.7",
        "matplotlib>=3.8.0",
        "pyspark>=3.5.0",
        "evaluate>=0.4.2",
    ],
    entry_points={
        "console_scripts": [
            "fhirformer=fhirformer.cli:run",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
