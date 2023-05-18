#!/usr/bin/env python
"""Utility functions for validating JSON objects against schemas."""
import json
import os
from typing import Union

import jsonschema


def validate_against_schema(data: dict, filename: Union[str, os.PathLike]):
    """Validate json-type data against a schema.

    :param data: dictionary with the parameters to be validated
    :type data: dict
    :param filename: name or path of the schema to validate against
    :type filename: Union[str, os.PathLike]
    """

    with open(filename, encoding="utf8") as handler:
        schema = json.load(handler)

    jsonschema.validate(schema=schema, instance=data)
