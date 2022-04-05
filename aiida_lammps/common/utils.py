"""Utility functions for the handling of the input files"""
from collections.abc import Iterable
from datetime import datetime

from dateutil.parser import parse as get_date


def generate_header(value: str) -> str:
    """
    Generate the header for the blocks.

    :param value: string indicating the input block
    :type value: str
    :return: header/footer for the input block
    :rtype: str
    """
    return "#" + value.center(80, "-") + "#\n"


def flatten(full_list: list) -> list:
    """Flattens a list of list into a flat list.

    :param full_list: list of lists to be flattened
    :type full_list: list
    :yield: flattened list
    :rtype: list
    """
    for element in full_list:
        if isinstance(element, Iterable) and not isinstance(element, (str, bytes)):
            yield from flatten(element)
        else:
            yield element


def convert_date_string(string):
    """converts date string e.g. '10 Nov 2017' to datetime object
    if None, return todays date
    '"""
    if string is None:
        date = datetime.today()
    else:
        date = get_date(string)
    return date


def convert_to_str(value):
    """convert True/False to yes/no and all values to strings"""
    if isinstance(value, bool):
        if value:
            return "yes"
        return "no"
    return str(value)


def _convert_values(value):
    if isinstance(value, (tuple, list)):
        return " ".join([convert_to_str(v) for v in value])
    return convert_to_str(value)


def join_keywords(dct, ignore=None):
    """join a dict of {keyword: value, ...} into a string 'keyword value ...'

    value can be a single value or a list/tuple of values
    """
    ignore = [] if not ignore else ignore
    return " ".join(
        [
            f"{k} {_convert_values(dct[k])}"
            for k in sorted(dct.keys())
            if k not in ignore
        ]
    )


def get_path(dct, path, default=None, raise_error=True):
    """return the value from a key path in a nested dictionary"""
    subdct = dct
    for i, key in enumerate(path):
        if not isinstance(subdct, dict) or key not in subdct:
            if raise_error:
                raise KeyError(f"path does not exist in dct: {path[0:i + 1]}")
            return default
        subdct = subdct[key]
    return subdct
