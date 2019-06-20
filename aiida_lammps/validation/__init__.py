import json
import os

import jsonschema


def read_schema(name):
    """read and return an json schema

    :param name: <name>.schema.json

    :return: dictionary
    """
    dirpath = os.path.dirname(os.path.realpath(__file__))
    jpath = os.path.join(dirpath, "{}.schema.json".format(name))
    with open(jpath) as jfile:
        schema = json.load(jfile)
    return schema


def validate_with_json(data, name):
    """ validate json-type data against a schema

    :param data: dictionary
    :param name: <name>.schema.json
    """
    schema = read_schema(name)
    validator = jsonschema.Draft4Validator

    # by default, only validates lists
    validator(schema, types={"array": (list, tuple)}).validate(data)


def validate_with_dict(data, schema):
    """ validate json-type data against a schema

    :param data: dictionary
    :param schema: dictionary
    """
    validator = jsonschema.Draft4Validator

    # by default, only validates lists
    validator(schema, types={"array": (list, tuple)}).validate(data)
