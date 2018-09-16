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