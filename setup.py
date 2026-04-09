"""
setup.py - Package configuration for the MLOps pipeline.

Install in editable mode for local development:
    pip install -e .

This makes all src/ modules importable as:
    from mlops_pipeline.data_ingestion import ingest
    from mlops_pipeline.train import train
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="mlops-pipeline",
    version="1.0.0",
    author="Jose",
    description="End-to-end MLOps pipeline: data versioning, experiment tracking, and automated deployment.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.11",

    # Treat src/ as the package root
    package_dir={"": "src"},
    packages=find_packages(where="src"),

    install_requires=requirements,

    extras_require={
        "api": [
            "fastapi>=0.109.0",
            "uvicorn>=0.27.0",
            "pydantic>=2.6.0",
        ],
        "dev": [
            "pytest>=7.4.4",
            "pytest-cov>=4.1.0",
            "flake8>=7.0.0",
            "black>=24.0.0",
            "httpx>=0.27.0",     # for smoke tests
        ],
    },

    entry_points={
        "console_scripts": [
            # Run full DVC pipeline
            "mlops-train=train:main",
            # CLI inference
            "mlops-predict=predict:main",
        ],
    },

    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
