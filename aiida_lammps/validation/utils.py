#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions for validating JSON objects against schemas."""
import io
import json
import os

import importlib_resources
import jsonschema

from aiida_lammps.validation import schemas


def load_schema(name):
    """Read and return a JSON schema.

    If the name is an absolute path, it will be used as is, otherwise
    it will be loaded as resource from the internal json schema module.

    Parameters
    ----------
    name: str

    Returns
    -------
    dict

    """
    if os.path.isabs(name):
        with io.open(name) as jfile:
            schema = json.load(jfile)
    else:
        schema = json.loads(importlib_resources.read_text(schemas, name))

    return schema


def load_validator(schema):
    """Create a validator for a schema.

    Parameters
    ----------
    schema : str or dict
        schema or path to schema

    Returns
    -------
    jsonschema.IValidator
        the validator to use

    """
    if isinstance(schema, str):
        schema = load_schema(schema)

    validator_cls = jsonschema.validators.validator_for(schema)
    validator_cls.check_schema(schema)

    # by default, only validates lists
    def is_array(checker, instance):
        return isinstance(instance, (tuple, list))

    type_checker = validator_cls.TYPE_CHECKER.redefine("array", is_array)
    validator_cls = jsonschema.validators.extend(
        validator_cls, type_checker=type_checker
    )

    validator = validator_cls(schema=schema)
    return validator


def validate_against_schema(data, schema):
    """Validate json-type data against a schema.

    Parameters
    ----------
    data: dict
    schema: dict or str
        schema, name of schema resource, or absolute path to a schema

    Raises
    ------
    jsonschema.exceptions.SchemaError
        if the schema is invalid
    jsonschema.exceptions.ValidationError
        if the instance is invalid

    Returns
    -------
    bool
        return True if validated


    """
    validator = load_validator(schema)
    # validator.validate(data)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    if errors:
        raise jsonschema.ValidationError(
            "\n".join(
                [
                    "- {} [key path: '{}']".format(
                        error.message, "/".join([str(p) for p in error.path])
                    )
                    for error in errors
                ]
            )
        )

    return True
