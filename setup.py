# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fluxgraph",
    version="0.0.2",  # bump version to avoid re-upload conflict
    author="ihtesham-jahangir",
    author_email="ceo@alphanetwork.com.pk",
    description="A lightweight Python framework for building, orchestrating, and deploying Agentic AI systems (MVP).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ihtesham-jahangir/fluxgraph",
    packages=find_packages(where="."),  # look in the current directory
    package_dir={"": "."},              # root is current dir
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'flux=fluxgraph.core.app:main',  # Register 'flux' command
        ],
    },
)
