from datetime import datetime

from dateutil.parser import parse as get_date


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
    """convert True/False to yes/no and all values to strongs"""
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
    return " ".join(["{0} {1}".format(k, _convert_values(v)) for k, v in dct.items()
                     if k not in ignore])


def get_path(dct, path, default=None, raise_error=True):
    """return the value from a key path in a nested dictionary"""
    subdct = dct
    for i, key in enumerate(path):
        if not isinstance(subdct, dict) or key not in subdct:
            if raise_error:
                raise KeyError("path does not exist in dct: {}".format(path[0:i+1]))
            else:
                return default
        subdct = subdct[key]
    return subdct
