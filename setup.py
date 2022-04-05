#!/usr/bin/env python
"""Define the setup for the `aiida-lammps` plugin."""
import json

from setuptools import find_packages, setup

if __name__ == "__main__":
    FILENAME_SETUP_JSON = "setup.json"
    FILENAME_DESCRIPTION = "README.md"

    with open(FILENAME_SETUP_JSON, "r") as handle:
        setup_json = json.load(handle)

    with open(FILENAME_DESCRIPTION, "r") as handle:
        description = handle.read()

    setup(
        packages=find_packages(),
        long_description=description,
        long_description_content_type="text/markdown",
        **setup_json,
    )
